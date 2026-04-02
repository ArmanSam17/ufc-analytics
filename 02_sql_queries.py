"""
02_sql_queries.py
-----------------
Runs 6 advanced SQL queries against ufc.db to demonstrate portfolio-level
SQL skills: window functions, CTEs, multi-table joins, and aggregations.
"""

import sqlite3
import textwrap
import pandas as pd

DB_PATH = "ufc.db"

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 30)

conn = sqlite3.connect(DB_PATH)


def run_query(num, title, description, sql):
    """Execute a query, print a formatted header, and display the result."""
    print("\n" + "=" * 80)
    print(f"  QUERY {num}: {title}")
    print(f"  {description}")
    print("=" * 80)
    df = pd.read_sql(textwrap.dedent(sql), conn)
    print(df.to_string(index=False))
    print(f"\n  [{len(df)} rows returned]")
    return df


# ---------------------------------------------------------------------------
# Query 1 — WINDOW FUNCTION
# Rank the top-10 finishers per weight class (min 5 wins).
# Finish = KO/TKO, Submission, or TKO - Doctor's Stoppage.
# RANK() OVER (PARTITION BY weight_class ORDER BY finish_rate DESC)
# ---------------------------------------------------------------------------

run_query(
    num=1,
    title="Top 10 Finishers per Weight Class (Window Function)",
    description="RANK() OVER (PARTITION BY weight_class ORDER BY finish_rate DESC) — min 5 wins",
    sql="""
        WITH fighter_results AS (
            -- Treat Red-corner results
            SELECT r_fighter_name AS fighter_name,
                   weight_class,
                   CASE WHEN winner = 'Red' THEN 1 ELSE 0 END                    AS is_win,
                   CASE WHEN winner = 'Red'
                         AND win_by IN ('KO/TKO','Submission','TKO - Doctor''s Stoppage')
                        THEN 1 ELSE 0 END                                         AS is_finish
            FROM bouts WHERE weight_class IS NOT NULL
            UNION ALL
            -- Blue-corner results
            SELECT b_fighter_name,
                   weight_class,
                   CASE WHEN winner = 'Blue' THEN 1 ELSE 0 END,
                   CASE WHEN winner = 'Blue'
                         AND win_by IN ('KO/TKO','Submission','TKO - Doctor''s Stoppage')
                        THEN 1 ELSE 0 END
            FROM bouts WHERE weight_class IS NOT NULL
        ),
        fighter_stats AS (
            SELECT fighter_name,
                   weight_class,
                   SUM(is_win)                                            AS total_wins,
                   SUM(is_finish)                                         AS finish_wins,
                   ROUND(CAST(SUM(is_finish) AS REAL) / SUM(is_win), 3)  AS finish_rate
            FROM fighter_results
            GROUP BY fighter_name, weight_class
            HAVING SUM(is_win) >= 5
        ),
        ranked AS (
            SELECT fighter_name,
                   weight_class,
                   total_wins,
                   finish_wins,
                   finish_rate,
                   RANK() OVER (
                       PARTITION BY weight_class
                       ORDER BY finish_rate DESC, finish_wins DESC
                   ) AS rank_in_class
            FROM fighter_stats
        )
        SELECT weight_class,
               rank_in_class  AS rank,
               fighter_name,
               total_wins,
               finish_wins,
               finish_rate
        FROM ranked
        WHERE rank_in_class <= 10
        ORDER BY weight_class, rank_in_class
    """,
)

# ---------------------------------------------------------------------------
# Query 2 — CTE
# Current win streak per fighter: consecutive wins at the END of their
# recorded career (most-recent fight first).
# Strategy: number fights newest-to-oldest; the streak length equals the
# row-number of the first non-win, minus 1 (or total fights if unbeaten).
# ---------------------------------------------------------------------------

run_query(
    num=2,
    title="Top 20 Current Win Streaks (CTE)",
    description="Consecutive wins ending at the fighter's most recent bout",
    sql="""
        WITH all_fights AS (
            -- Unify Red- and Blue-corner appearances with a win/loss label
            SELECT r_fighter_name AS fighter_name,
                   e.event_date,
                   CASE WHEN winner = 'Red' THEN 'W' ELSE 'L' END AS result
            FROM bouts b JOIN events e USING (event_id)
            UNION ALL
            SELECT b_fighter_name,
                   e.event_date,
                   CASE WHEN winner = 'Blue' THEN 'W' ELSE 'L' END
            FROM bouts b JOIN events e USING (event_id)
        ),
        numbered AS (
            -- Rank each fight newest-first so rn=1 is the most recent bout
            SELECT fighter_name,
                   event_date,
                   result,
                   ROW_NUMBER() OVER (
                       PARTITION BY fighter_name
                       ORDER BY event_date DESC
                   ) AS rn
            FROM all_fights
        ),
        first_loss AS (
            -- The earliest row-number (from the top) where the streak breaks
            SELECT fighter_name,
                   MIN(rn) AS break_at
            FROM numbered
            WHERE result = 'L'
            GROUP BY fighter_name
        ),
        total_fights AS (
            SELECT fighter_name, MAX(rn) AS n_fights
            FROM numbered
            GROUP BY fighter_name
        ),
        streaks AS (
            SELECT tf.fighter_name,
                   tf.n_fights,
                   -- If never lost: streak = all fights; else streak = break_at - 1
                   COALESCE(fl.break_at - 1, tf.n_fights) AS win_streak
            FROM total_fights tf
            LEFT JOIN first_loss fl USING (fighter_name)
            WHERE tf.n_fights >= 3     -- exclude fighters with very few bouts
        )
        SELECT fighter_name,
               win_streak,
               n_fights                           AS total_bouts
        FROM streaks
        ORDER BY win_streak DESC, n_fights DESC
        LIMIT 20
    """,
)

# ---------------------------------------------------------------------------
# Query 3 — AGGREGATION + JOIN
# Per weight class: avg fight duration (seconds), avg significant strikes
# landed per fight (summing both corners), finish rate, total bouts.
# Fight duration = completed rounds × 300 s + final-round time seconds.
# ---------------------------------------------------------------------------

run_query(
    num=3,
    title="Fight Statistics by Weight Class (Aggregation + Join)",
    description="Avg duration, avg sig. strikes, finish rate — ordered by total fights",
    sql="""
        WITH fight_duration AS (
            -- Convert last_round + last_round_time ("M:SS") to total seconds
            SELECT bout_id,
                   (last_round - 1) * 300
                   + CAST(SUBSTR(last_round_time, 1,
                          INSTR(last_round_time, ':') - 1) AS INTEGER) * 60
                   + CAST(SUBSTR(last_round_time,
                          INSTR(last_round_time, ':') + 1) AS INTEGER) AS duration_sec
            FROM bouts
            WHERE last_round_time IS NOT NULL AND last_round_time != '---'
        ),
        sig_str_per_fight AS (
            -- Sum strikes from both corners for each bout
            SELECT bout_id,
                   SUM(sig_str_landed) AS total_sig_str
            FROM bout_stats
            GROUP BY bout_id
        )
        SELECT b.weight_class,
               COUNT(*)                                          AS total_fights,
               ROUND(AVG(fd.duration_sec), 0)                   AS avg_duration_sec,
               ROUND(AVG(ss.total_sig_str), 1)                  AS avg_sig_str_per_fight,
               ROUND(
                   100.0 * SUM(CASE WHEN b.win_by IN (
                       'KO/TKO','Submission','TKO - Doctor''s Stoppage'
                   ) THEN 1 ELSE 0 END) / COUNT(*), 1
               )                                                 AS finish_rate_pct
        FROM bouts b
        LEFT JOIN fight_duration fd USING (bout_id)
        LEFT JOIN sig_str_per_fight ss USING (bout_id)
        WHERE b.weight_class IS NOT NULL
        GROUP BY b.weight_class
        ORDER BY total_fights DESC
    """,
)

# ---------------------------------------------------------------------------
# Query 4 — LAG WINDOW FUNCTION
# Days of rest between fights per fighter — average gap between consecutive
# bouts, using LAG() over event_date ordered chronologically.
# Return fighters with 5+ fights ordered by least rest (busiest schedule).
# ---------------------------------------------------------------------------

run_query(
    num=4,
    title="Average Days Rest Between Fights per Fighter (LAG Window Function)",
    description="LAG(event_date) OVER (PARTITION BY fighter ORDER BY date) — fighters with 5+ bouts",
    sql="""
        WITH all_appearances AS (
            -- One row per unique fighter-date appearance
            SELECT DISTINCT
                   r_fighter_name AS fighter_name, e.event_date
            FROM bouts b JOIN events e USING (event_id)
            UNION
            SELECT DISTINCT
                   b_fighter_name, e.event_date
            FROM bouts b JOIN events e USING (event_id)
        ),
        with_prev AS (
            -- Attach the previous fight date using LAG()
            SELECT fighter_name,
                   event_date,
                   LAG(event_date) OVER (
                       PARTITION BY fighter_name
                       ORDER BY event_date
                   ) AS prev_fight_date
            FROM all_appearances
        ),
        gaps AS (
            SELECT fighter_name,
                   JULIANDAY(event_date) - JULIANDAY(prev_fight_date) AS days_rest
            FROM with_prev
            WHERE prev_fight_date IS NOT NULL
        )
        SELECT fighter_name,
               COUNT(*)                         AS gaps_counted,
               ROUND(AVG(days_rest), 0)         AS avg_days_rest,
               MIN(CAST(days_rest AS INTEGER))  AS min_days_rest,
               MAX(CAST(days_rest AS INTEGER))  AS max_days_rest
        FROM gaps
        GROUP BY fighter_name
        HAVING COUNT(*) >= 4       -- 4 gaps = at least 5 fights
        ORDER BY avg_days_rest ASC
        LIMIT 25
    """,
)

# ---------------------------------------------------------------------------
# Query 5 — MULTI-TABLE JOIN
# "Tale of the tape" for every title fight: both fighters' height, reach,
# stance, and the fight result. Uses two LEFT JOINs to the fighters table.
# ---------------------------------------------------------------------------

run_query(
    num=5,
    title="Title Fight Tale of the Tape (Multi-Table Join)",
    description="Joins bouts → events → fighters (x2) for every title_bout = 1",
    sql="""
        SELECT e.event_date,
               b.weight_class,
               b.r_fighter_name                             AS red_fighter,
               ROUND(rf.height_cm, 1)                      AS red_ht_cm,
               ROUND(rf.reach_cm, 1)                        AS red_reach_cm,
               rf.stance                                    AS red_stance,
               b.b_fighter_name                             AS blue_fighter,
               ROUND(bf.height_cm, 1)                      AS blue_ht_cm,
               ROUND(bf.reach_cm, 1)                        AS blue_reach_cm,
               bf.stance                                    AS blue_stance,
               b.winner,
               b.win_by
        FROM bouts b
        JOIN  events  e  USING (event_id)
        LEFT JOIN fighters rf ON b.r_fighter_name = rf.fighter_name
        LEFT JOIN fighters bf ON b.b_fighter_name = bf.fighter_name
        WHERE b.title_bout = 1
        ORDER BY e.event_date DESC
        LIMIT 20
    """,
)

# ---------------------------------------------------------------------------
# Query 6 — CTE CHAIN
# CTE 1 — Pull per-fighter striking and grappling volume from the fighters
#          table (career averages), filtered to fighters with enough data.
# CTE 2 — Classify each fighter into a style archetype using those metrics:
#            • Grappler        : high TD or sub volume  (td_avg ≥ 2 OR sub_avg ≥ 1)
#            • Pressure Fighter: high output + accuracy  (slpm ≥ 4 AND str_acc ≥ 0.45)
#            • Counter Striker : absorbs less than output + good defence (slpm > sapm AND str_def ≥ 0.55)
#            • Balanced        : everything else
# Final SELECT — count and average finish rate per style.
# ---------------------------------------------------------------------------

run_query(
    num=6,
    title="Fighter Style Classification & Finish Rate (CTE Chain)",
    description="CTE 1: metrics from fighters table → CTE 2: style label → aggregate finish rate",
    sql="""
        WITH fighter_metrics AS (
            -- Pull career-average stats; keep only fighters with complete data
            SELECT fighter_name, slpm, str_acc, sapm, str_def, td_avg, sub_avg
            FROM fighters
            WHERE slpm   IS NOT NULL AND str_acc IS NOT NULL
              AND sapm   IS NOT NULL AND str_def IS NOT NULL
              AND td_avg IS NOT NULL AND sub_avg IS NOT NULL
              AND (slpm + str_acc + td_avg) > 0    -- exclude zeroed-out rows
        ),
        fighter_finish_rates AS (
            -- Compute finish rate for fighters with at least 5 wins
            SELECT fighter_name,
                   SUM(is_win)     AS wins,
                   SUM(is_finish)  AS finishes,
                   ROUND(
                       CAST(SUM(is_finish) AS REAL) / NULLIF(SUM(is_win), 0),
                   3) AS finish_rate
            FROM (
                SELECT r_fighter_name AS fighter_name,
                       CASE WHEN winner = 'Red' THEN 1 ELSE 0 END AS is_win,
                       CASE WHEN winner = 'Red'
                             AND win_by IN ('KO/TKO','Submission','TKO - Doctor''s Stoppage')
                            THEN 1 ELSE 0 END                      AS is_finish
                FROM bouts
                UNION ALL
                SELECT b_fighter_name,
                       CASE WHEN winner = 'Blue' THEN 1 ELSE 0 END,
                       CASE WHEN winner = 'Blue'
                             AND win_by IN ('KO/TKO','Submission','TKO - Doctor''s Stoppage')
                            THEN 1 ELSE 0 END
                FROM bouts
            ) raw
            GROUP BY fighter_name
            HAVING SUM(is_win) >= 5
        ),
        style_classified AS (
            -- Apply classification rules in priority order
            SELECT fm.fighter_name,
                   ffr.wins,
                   ffr.finish_rate,
                   CASE
                       WHEN fm.td_avg >= 2.0 OR fm.sub_avg >= 1.0          THEN 'Grappler'
                       WHEN fm.slpm   >= 4.0 AND fm.str_acc >= 0.45        THEN 'Pressure Fighter'
                       WHEN fm.slpm   >  fm.sapm AND fm.str_def >= 0.55    THEN 'Counter Striker'
                       ELSE 'Balanced'
                   END AS style
            FROM fighter_metrics fm
            JOIN fighter_finish_rates ffr USING (fighter_name)
        )
        SELECT style,
               COUNT(*)                        AS fighter_count,
               ROUND(AVG(finish_rate), 3)      AS avg_finish_rate,
               ROUND(AVG(wins), 1)             AS avg_wins
        FROM style_classified
        GROUP BY style
        ORDER BY avg_finish_rate DESC
    """,
)

conn.close()
print("\n" + "=" * 80)
print("  All 6 queries complete.")
print("=" * 80)
