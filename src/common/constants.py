# coding:utf-8
import os
import json

raw_special_tokens = json.load(
    open(f"{os.path.dirname(__file__)}/additional-tokens.json", "r", encoding="utf-8")
)
special_tokens = [itm.lstrip("Ġ") for itm in raw_special_tokens]

recategorizations = [
    "\u0120COUNTRY",
    "\u0120QUANTITY",
    "\u0120ORGANIZATION",
    "\u0120DATE_ATTRS",
    "\u0120NATIONALITY",
    "\u0120LOCATION",
    "\u0120ENTITY",
    "\u0120MISC",
    "\u0120ORDINAL_ENTITY",
    "\u0120IDEOLOGY",
    "\u0120RELIGION",
    "\u0120STATE_OR_PROVINCE",
    "\u0120CAUSE_OF_DEATH",
    "\u0120TITLE",
    "\u0120DATE",
    "\u0120NUMBER",
    "\u0120HANDLE",
    "\u0120SCORE_ENTITY",
    "\u0120DURATION",
    "\u0120ORDINAL",
    "\u0120MONEY",
    "\u0120CRIMINAL_CHARGE",
]

# special_tokens = ["<AMR>", "</AMR>"]