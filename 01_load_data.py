"""
01_load_data.py
---------------
Creates ufc.db (SQLite) with 4 tables and loads the raw CSV data into them.

Tables:
  fighters   -- physical attributes and career stats per fighter
  events     -- unique event (date + location) combinations
  bouts      -- one row per fight with result metadata
  bout_stats -- per-corner striking/grappling stats for each fight
"""

import sqlite3
import re
import pandas as pd

DB_PATH = "ufc.db"

# ---------------------------------------------------------------------------
# Helper functions for cleaning raw string formats
# ---------------------------------------------------------------------------

def parse_of_string(s):
    """Parse 'X of Y' strings -> (landed int, attempted int). Returns (None, None) on failure."""
    if pd.isna(s) or str(s).strip() in ("---", ""):
        return None, None
    m = re.match(r"(\d+)\s+of\s+(\d+)", str(s).strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_ctrl_time(s):
    """Parse 'M:SS' control-time string -> total seconds as int. Returns None on failure."""
    if pd.isna(s) or str(s).strip() in ("---", ""):
        return None
    m = re.match(r"(\d+):(\d+)", str(s).strip())
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def parse_pct(s):
    """Parse '39%' -> 0.39. Returns None on failure."""
    if pd.isna(s) or str(s).strip() in ("---", ""):
        return None
    m = re.match(r"([\d.]+)%", str(s).strip())
    if m:
        return float(m.group(1)) / 100.0
    return None


def parse_height_cm(s):
    """Parse '5' 11"' -> centimetres as float. Returns None on failure."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    m = re.match(r"(\d+)'\s*(\d+)", str(s).strip())
    if m:
        inches = int(m.group(1)) * 12 + int(m.group(2))
        return round(inches * 2.54, 2)
    return None


def parse_weight_lbs(s):
    """Parse '155 lbs.' -> 155.0. Returns None on failure."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    m = re.match(r"([\d.]+)", str(s).strip())
    if m:
        return float(m.group(1))
    return None


def parse_reach_cm(s):
    """Parse '76"' -> centimetres as float. Returns None on failure."""
    if pd.isna(s) or str(s).strip() == "":
        return None
    m = re.match(r"([\d.]+)", str(s).strip())
    if m:
        return round(float(m.group(1)) * 2.54, 2)
    return None


# ---------------------------------------------------------------------------
# Load raw CSVs
# ---------------------------------------------------------------------------

print("Loading raw CSV files...")

raw_fighters = pd.read_csv("data/raw_fighter_details.csv")
# raw_total_fight_data uses semicolon as delimiter
raw_fights = pd.read_csv("data/raw_total_fight_data.csv", sep=";")
# data.csv enriches fights with title_bout, weight_class, and pre-fight averages
enriched = pd.read_csv("data/data.csv")

print(f"  raw_fighter_details : {len(raw_fighters):,} rows")
print(f"  raw_total_fight_data: {len(raw_fights):,} rows")
print(f"  data                : {len(enriched):,} rows")

# ---------------------------------------------------------------------------
# Build the fighters DataFrame
# ---------------------------------------------------------------------------

print("\nCleaning fighters data...")

fighters_df = raw_fighters.copy()
fighters_df = fighters_df.drop_duplicates(subset="fighter_name")

fighters_df["height_cm"]  = fighters_df["Height"].apply(parse_height_cm)
fighters_df["weight_lbs"] = fighters_df["Weight"].apply(parse_weight_lbs)
fighters_df["reach_cm"]   = fighters_df["Reach"].apply(parse_reach_cm)
fighters_df["str_acc"]    = fighters_df["Str_Acc"].apply(parse_pct)
fighters_df["sapm"]       = fighters_df["SApM"]
fighters_df["str_def"]    = fighters_df["Str_Def"].apply(parse_pct)
fighters_df["td_acc"]     = fighters_df["TD_Acc"].apply(parse_pct)
fighters_df["td_def"]     = fighters_df["TD_Def"].apply(parse_pct)

# Parse DOB to ISO date string (YYYY-MM-DD)
fighters_df["dob"] = pd.to_datetime(fighters_df["DOB"], format="%b %d, %Y", errors="coerce").dt.strftime("%Y-%m-%d")

fighters_clean = fighters_df[[
    "fighter_name", "height_cm", "weight_lbs", "reach_cm", "Stance", "dob",
    "SLpM", "str_acc", "sapm", "str_def",
    "TD_Avg", "td_acc", "td_def", "Sub_Avg",
]].rename(columns={
    "Stance":  "stance",
    "SLpM":    "slpm",
    "TD_Avg":  "td_avg",
    "Sub_Avg": "sub_avg",
})

# ---------------------------------------------------------------------------
# Build the events DataFrame  (unique date + location)
# ---------------------------------------------------------------------------

print("Building events data...")

# Parse fight dates to ISO format
raw_fights["event_date"] = pd.to_datetime(raw_fights["date"], format="%B %d, %Y", errors="coerce").dt.strftime("%Y-%m-%d")

events_df = (
    raw_fights[["event_date", "location"]]
    .drop_duplicates()
    .dropna(subset=["event_date"])
    .reset_index(drop=True)
)
# Assign a surrogate key — index+1 so it's 1-based
events_df.insert(0, "event_id", events_df.index + 1)

# ---------------------------------------------------------------------------
# Build the bouts DataFrame
# ---------------------------------------------------------------------------

print("Building bouts data...")

# Add title_bout and weight_class from enriched data.csv by matching on
# (R_fighter, B_fighter, date).  We normalise the date format first.
enriched["event_date"] = pd.to_datetime(enriched["date"], errors="coerce").dt.strftime("%Y-%m-%d")
enriched_meta = enriched[["R_fighter", "B_fighter", "event_date", "title_bout", "weight_class"]].copy()

raw_fights_aug = raw_fights.merge(
    enriched_meta,
    on=["R_fighter", "B_fighter", "event_date"],
    how="left",
)

# Join event_id onto each fight row
bouts_df = raw_fights_aug.merge(
    events_df[["event_id", "event_date", "location"]],
    on=["event_date", "location"],
    how="left",
)

# Normalise the Winner column: the raw CSV stores the fighter name;
# map to 'Red', 'Blue', 'Draw', or 'NC' so it matches enriched data.
def normalise_winner(row):
    w = row.get("Winner", None)
    if pd.isna(w) or w == "":
        return None
    if w == row["R_fighter"]:
        return "Red"
    if w == row["B_fighter"]:
        return "Blue"
    return str(w)   # 'Draw' or 'NC' already present as literals

bouts_df["winner"] = bouts_df.apply(normalise_winner, axis=1)

# Cast booleans so SQLite stores 0/1
bouts_df["title_bout"] = bouts_df["title_bout"].map(
    {True: 1, False: 0, "True": 1, "False": 0}
).fillna(0).astype(int)

bouts_df = bouts_df.reset_index(drop=True)
bouts_df.insert(0, "bout_id", bouts_df.index + 1)

bouts_clean = bouts_df[[
    "bout_id", "event_id",
    "R_fighter", "B_fighter",
    "winner", "win_by", "last_round", "last_round_time",
    "Format", "Referee", "Fight_type",
    "title_bout", "weight_class",
]].rename(columns={
    "R_fighter":       "r_fighter_name",
    "B_fighter":       "b_fighter_name",
    "Format":          "bout_format",
    "Referee":         "referee",
    "Fight_type":      "fight_type",
})

# ---------------------------------------------------------------------------
# Build the bout_stats DataFrame  (two rows per fight: Red corner + Blue corner)
# ---------------------------------------------------------------------------

print("Building bout_stats data...")

def build_corner_stats(df, corner):
    """Extract per-corner stats from the wide fight DataFrame."""
    p = corner[0].upper()   # 'R' or 'B'

    sig_landed, sig_att   = zip(*df[f"{p}_SIG_STR."].apply(parse_of_string))
    tot_landed, tot_att   = zip(*df[f"{p}_TOTAL_STR."].apply(parse_of_string))
    td_landed,  td_att    = zip(*df[f"{p}_TD"].apply(parse_of_string))
    head_landed, head_att = zip(*df[f"{p}_HEAD"].apply(parse_of_string))
    body_landed, body_att = zip(*df[f"{p}_BODY"].apply(parse_of_string))
    leg_landed,  leg_att  = zip(*df[f"{p}_LEG"].apply(parse_of_string))
    dist_landed, dist_att = zip(*df[f"{p}_DISTANCE"].apply(parse_of_string))
    cln_landed,  cln_att  = zip(*df[f"{p}_CLINCH"].apply(parse_of_string))
    gnd_landed,  gnd_att  = zip(*df[f"{p}_GROUND"].apply(parse_of_string))

    return pd.DataFrame({
        "bout_id":          df["bout_id"].values,
        "corner":           corner,
        "kd":               pd.to_numeric(df[f"{p}_KD"], errors="coerce"),
        "sig_str_landed":   list(sig_landed),
        "sig_str_att":      list(sig_att),
        "sig_str_pct":      df[f"{p}_SIG_STR_pct"].apply(parse_pct).values,
        "total_str_landed": list(tot_landed),
        "total_str_att":    list(tot_att),
        "td_landed":        list(td_landed),
        "td_att":           list(td_att),
        "td_pct":           df[f"{p}_TD_pct"].apply(parse_pct).values,
        "sub_att":          pd.to_numeric(df[f"{p}_SUB_ATT"], errors="coerce"),
        "reversals":        pd.to_numeric(df[f"{p}_REV"], errors="coerce"),
        "ctrl_time_sec":    df[f"{p}_CTRL"].apply(parse_ctrl_time).values,
        "head_landed":      list(head_landed),
        "head_att":         list(head_att),
        "body_landed":      list(body_landed),
        "body_att":         list(body_att),
        "leg_landed":       list(leg_landed),
        "leg_att":          list(leg_att),
        "distance_landed":  list(dist_landed),
        "distance_att":     list(dist_att),
        "clinch_landed":    list(cln_landed),
        "clinch_att":       list(cln_att),
        "ground_landed":    list(gnd_landed),
        "ground_att":       list(gnd_att),
    })

red_stats  = build_corner_stats(bouts_df, "Red")
blue_stats = build_corner_stats(bouts_df, "Blue")

bout_stats_df = (
    pd.concat([red_stats, blue_stats], ignore_index=True)
    .sort_values("bout_id")
    .reset_index(drop=True)
)
bout_stats_df.insert(0, "stat_id", bout_stats_df.index + 1)

# ---------------------------------------------------------------------------
# Create SQLite database and tables
# ---------------------------------------------------------------------------

print(f"\nCreating SQLite database: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")
cur = conn.cursor()

# fighters table
cur.executescript("""
DROP TABLE IF EXISTS bout_stats;
DROP TABLE IF EXISTS bouts;
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS fighters;

CREATE TABLE fighters (
    fighter_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    fighter_name TEXT    NOT NULL UNIQUE,
    height_cm    REAL,
    weight_lbs   REAL,
    reach_cm     REAL,
    stance       TEXT,
    dob          TEXT,           -- ISO date string YYYY-MM-DD
    slpm         REAL,           -- significant strikes landed per minute
    str_acc      REAL,           -- striking accuracy (0-1)
    sapm         REAL,           -- significant strikes absorbed per minute
    str_def      REAL,           -- strike defence rate (0-1)
    td_avg       REAL,           -- average takedowns landed per 15 min
    td_acc       REAL,           -- takedown accuracy (0-1)
    td_def       REAL,           -- takedown defence rate (0-1)
    sub_avg      REAL            -- average submission attempts per 15 min
);

CREATE TABLE events (
    event_id   INTEGER PRIMARY KEY,
    event_date TEXT NOT NULL,    -- ISO date string YYYY-MM-DD
    location   TEXT NOT NULL
);

CREATE TABLE bouts (
    bout_id        INTEGER PRIMARY KEY,
    event_id       INTEGER NOT NULL REFERENCES events(event_id),
    r_fighter_name TEXT    NOT NULL,
    b_fighter_name TEXT    NOT NULL,
    winner         TEXT,         -- 'Red', 'Blue', 'Draw', 'NC', or NULL
    win_by         TEXT,
    last_round     INTEGER,
    last_round_time TEXT,
    bout_format    TEXT,
    referee        TEXT,
    fight_type     TEXT,
    title_bout     INTEGER NOT NULL DEFAULT 0,  -- 0/1 boolean
    weight_class   TEXT
);

CREATE TABLE bout_stats (
    stat_id          INTEGER PRIMARY KEY,
    bout_id          INTEGER NOT NULL REFERENCES bouts(bout_id),
    corner           TEXT    NOT NULL CHECK (corner IN ('Red', 'Blue')),
    kd               INTEGER,   -- knockdowns
    sig_str_landed   INTEGER,
    sig_str_att      INTEGER,
    sig_str_pct      REAL,
    total_str_landed INTEGER,
    total_str_att    INTEGER,
    td_landed        INTEGER,
    td_att           INTEGER,
    td_pct           REAL,
    sub_att          INTEGER,   -- submission attempts
    reversals        INTEGER,
    ctrl_time_sec    INTEGER,   -- control time in seconds
    head_landed      INTEGER,
    head_att         INTEGER,
    body_landed      INTEGER,
    body_att         INTEGER,
    leg_landed       INTEGER,
    leg_att          INTEGER,
    distance_landed  INTEGER,
    distance_att     INTEGER,
    clinch_landed    INTEGER,
    clinch_att       INTEGER,
    ground_landed    INTEGER,
    ground_att       INTEGER
);
""")
conn.commit()

# ---------------------------------------------------------------------------
# Load data into tables
# ---------------------------------------------------------------------------

print("Inserting fighters...")
fighters_clean.to_sql("fighters", conn, if_exists="append", index=False)

print("Inserting events...")
events_df.to_sql("events", conn, if_exists="append", index=False)

print("Inserting bouts...")
bouts_clean.to_sql("bouts", conn, if_exists="append", index=False)

print("Inserting bout_stats...")
bout_stats_df.to_sql("bout_stats", conn, if_exists="append", index=False)

conn.commit()

# ---------------------------------------------------------------------------
# Verification: print row counts
# ---------------------------------------------------------------------------

print("\n" + "=" * 45)
print("Row counts per table")
print("=" * 45)
for table in ("fighters", "events", "bouts", "bout_stats"):
    count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<12} : {count:>6,} rows")
print("=" * 45)

# Sanity checks
print("\nSample fighter:")
row = cur.execute("SELECT fighter_name, height_cm, weight_lbs, stance, dob FROM fighters LIMIT 1").fetchone()
print(f"  {row}")

print("\nSample bout:")
row = cur.execute("""
    SELECT b.bout_id, e.event_date, b.r_fighter_name, b.b_fighter_name,
           b.winner, b.win_by, b.weight_class, b.title_bout
    FROM bouts b JOIN events e USING (event_id)
    LIMIT 1
""").fetchone()
print(f"  {row}")

print("\nSample bout_stats (Red corner):")
row = cur.execute("""
    SELECT stat_id, bout_id, corner, kd, sig_str_landed, sig_str_att,
           td_landed, td_att, ctrl_time_sec
    FROM bout_stats WHERE corner = 'Red' LIMIT 1
""").fetchone()
print(f"  {row}")

conn.close()
print("\nDone. Database saved to", DB_PATH)
