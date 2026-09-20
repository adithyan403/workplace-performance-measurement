"""WPM Module 2: PROCESS -- process model, activity catalogue, zones,
thresholds and event registration (mirrors the paper's P_Process_2 module).

This module carries the domain knowledge of the analysed process:
  * what activities exist and which are value-added (paper's process model),
  * where the workspace areas are (primary writing zone / crew zone),
  * what motion thresholds separate activity from idleness.
"""
import json
from pathlib import Path

try:
    import pymongo
    import datetime
except Exception:
    pymongo = None


class ProcessModule:
    def __init__(self, cfg_file_or_dict, root="."):
        self.root = Path(root)
        if isinstance(cfg_file_or_dict, (str, Path)):
            with open(self.root / cfg_file_or_dict, "r", encoding="utf-8") as f:
                self.cfg = json.load(f)
        else:
            self.cfg = cfg_file_or_dict

    # ------------------------------------------------------------------ #
    # catalogue accessors                                                #
    # ------------------------------------------------------------------ #
    @property
    def activities(self):
        return {a["id"]: a for a in self.cfg["activities"]}

    @property
    def va_activities(self):
        return [a for a in self.cfg["activities"] if a["va"]]

    @property
    def nva_activities(self):
        return [a for a in self.cfg["activities"] if not a["va"]]

    def is_va(self, activity):
        a = self.activities.get(activity)
        return bool(a and a["va"])

    def color(self, activity):
        a = self.activities.get(activity)
        return a["color"] if a else "#94a3b8"

    # ------------------------------------------------------------------ #
    # digital trace -> event registration (paper: events in SQL, here Mongo)
    # ------------------------------------------------------------------ #
    def register_event(self, mongo_uri, event, collection="duty_logs"):
        if pymongo is None or not mongo_uri:
            raise RuntimeError("pymongo or mongo_uri missing")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
        db = client[self.cfg["db"]["name"]]
        db[collection].insert_one({**event, "logged_at": datetime.datetime.utcnow()})
        client.close()
        return True

    def load_event_log(self, mongo_uri, collection="duty_logs", limit=0):
        if pymongo is None or not mongo_uri:
            return []
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
        db = client[self.cfg["db"]["name"]]
        cur = db[collection].find({}, {"_id": 0}).sort("start_sec", 1)
        if limit:
            cur = cur.limit(limit)
        rows = list(cur)
        client.close()
        return rows