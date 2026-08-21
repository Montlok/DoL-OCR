# -*- coding: utf-8 -*-
"""Traditional Mongolian suffix inventory for tokenizer experiments."""

try:
    from .unicode_norm import NNBSP, strip_all
except ImportError:  # pragma: no cover - supports direct script execution.
    from unicode_norm import NNBSP, strip_all  # type: ignore[no-redef]


CASE_SUFFIXES = [
    {
        "id": "GEN",
        "name": "genitive",
        "surface": ["ᠤᠨ", "ᠦᠨ", "ᠶᠢᠨ", "ᠤ", "ᠦ", "ᠢᠨ", "ᠨᠤ", "ᠨᠦ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "of",
        "nnbsp": True,
    },
    {
        "id": "ACC",
        "name": "accusative",
        "surface": ["ᠶᠢ", "ᠢ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 2,
        "notes": "direct object",
        "nnbsp": True,
    },
    {
        "id": "DAT_LOC",
        "name": "dative-locative",
        "surface": [
            "ᠳᠤ",
            "ᠳᠦ",
            "ᠲᠤ",
            "ᠲᠦ",
            "ᠳᠤᠷ",
            "ᠳᠦᠷ",
            "ᠲᠤᠷ",
            "ᠲᠦᠷ",
            "ᠳᠠ",
            "ᠳᠡ",
            "ᠠ",
            "ᠡ",
        ],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "to/in/at",
        "nnbsp": True,
    },
    {
        "id": "LOC_ATTR",
        "name": "locative-attributive",
        "surface": ["ᠳᠠᠬᠢ", "ᠳᠡᠬᠢ", "ᠲᠠᠬᠢ", "ᠲᠡᠬᠢ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "in/at/on; historical DAQI/DEQI/TAQI/TEQI",
        "nnbsp": True,
    },
    {
        "id": "ABL",
        "name": "ablative",
        "surface": ["ᠠᠴᠠ", "ᠡᠴᠡ", "ᠴᠠ", "ᠴᠡ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "from",
        "nnbsp": True,
    },
    {
        "id": "INST",
        "name": "instrumental",
        "surface": ["ᠪᠠᠷ", "ᠪᠡᠷ", "ᠢᠶᠠᠷ", "ᠢᠶᠡᠷ", "ᠶᠠᠷ", "ᠶᠡᠷ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "by/with",
        "nnbsp": True,
    },
    {
        "id": "COM",
        "name": "comitative",
        "surface": ["ᠯᠤᠭ᠎ᠠ", "ᠯᠦᠭᠡ", "ᠲᠠᠢ", "ᠲᠡᠢ", "ᠲᠣᠢ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "with",
        "nnbsp": True,
    },
    {
        "id": "DIR",
        "name": "directive",
        "surface": ["ᠤᠷᠤᠭᠤ", "ᠦᠷᠦᠭᠦ", "ᠷᠤ", "ᠷᠦ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 2,
        "notes": "towards",
        "nnbsp": True,
    },
    {
        "id": "TERM",
        "name": "terminative",
        "surface": ["ᠬᠦᠷᠲᠡᠯ᠎ᠡ", "ᠬᠦᠷᠲᠡᠯᠡ"],
        "type": "case",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 2,
        "notes": "until",
        "nnbsp": True,
    },
]


PLURAL_SUFFIXES = [
    {
        "id": "PL_NAR",
        "name": "plural-nar",
        "surface": ["ᠨᠠᠷ", "ᠨᠡᠷ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "human",
        "nnbsp": True,
    },
    {
        "id": "PL_NUGUD",
        "name": "plural-nugud",
        "surface": ["ᠨᠤᠭᠤᠳ", "ᠨᠦᠭᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "general",
        "nnbsp": True,
    },
    {
        "id": "PL_UD",
        "name": "plural-ud",
        "surface": ["ᠤᠳ", "ᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "general",
        "nnbsp": True,
    },
    {
        "id": "PL_D",
        "name": "plural-d",
        "surface": ["ᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 1,
        "notes": "short",
        "nnbsp": True,
    },
    {
        "id": "PL_S",
        "name": "plural-s",
        "surface": ["ᠰ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 1,
        "notes": "short",
        "nnbsp": True,
    },
    {
        "id": "PL_CHUD",
        "name": "plural-chud",
        "surface": ["ᠴᠤᠳ", "ᠴᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "collective",
        "nnbsp": True,
    },
    {
        "id": "PL_DUD",
        "name": "plural-dud",
        "surface": ["ᠳᠤᠳ", "ᠳᠦᠳ"],
        "type": "plural",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 1,
        "notes": "collective",
        "nnbsp": True,
    },
]


POSSESSIVE_SUFFIXES = [
    {
        "id": "POSS_1SG",
        "name": "possessive-1sg",
        "surface": ["ᠮᠢᠨᠢ", "ᠮᠢᠨ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "my",
        "nnbsp": True,
    },
    {
        "id": "POSS_2SG",
        "name": "possessive-2sg",
        "surface": ["ᠴᠢᠨᠢ", "ᠴᠢᠨ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "your",
        "nnbsp": True,
    },
    {
        "id": "POSS_3",
        "name": "possessive-3",
        "surface": ["ᠨᠢ", "ᠢᠨᠤ", "ᠶᠢᠨᠤ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "his/her/its",
        "nnbsp": True,
    },
    {
        "id": "POSS_1PL",
        "name": "possessive-1pl",
        "surface": ["ᠮᠠᠨᠤ", "ᠮᠠᠨᠢ", "ᠪᠢᠳᠡᠨᠦ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "our",
        "nnbsp": True,
    },
    {
        "id": "POSS_2PL",
        "name": "possessive-2pl",
        "surface": ["ᠲᠠᠨᠤ", "ᠲᠠᠨᠢ"],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 3,
        "notes": "your-plural",
        "nnbsp": True,
    },
    {
        "id": "REFL_POSS",
        "name": "reflexive-possessive",
        "surface": [
            "ᠪᠠᠨ",
            "ᠪᠡᠨ",
            "ᠢᠶᠠᠨ",
            "ᠢᠶᠡᠨ",
            "ᠶᠠᠨ",
            "ᠶᠡᠨ",
            "ᠶᠤᠭᠠᠨ",
            "ᠶᠦᠭᠡᠨ",
            "ᠳᠠᠭᠠᠨ",
            "ᠳᠡᠭᠡᠨ",
            "ᠲᠠᠭᠠᠨ",
            "ᠲᠡᠭᠡᠨ",
            "ᠠᠴᠠᠭᠠᠨ",
            "ᠡᠴᠡᠭᠡᠨ",
            "ᠲᠠᠢᠭᠠᠨ",
            "ᠲᠡᠢᠭᠡᠨ",
        ],
        "type": "possessive",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 3,
        "notes": "own",
        "nnbsp": True,
    },
]


VOICE_SUFFIXES = [
    {
        "id": "CAUS_GUL",
        "name": "causative",
        "surface": ["ᠭᠤᠯ", "ᠭᠦᠯ", "ᠤᠯ", "ᠦᠯ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "causative",
    },
    {
        "id": "CAUS_GA",
        "name": "causative",
        "surface": ["ᠭᠠ", "ᠭᠡ", "ᠭᠠᠯ", "ᠭᠡᠯ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "causative",
    },
    {
        "id": "PASS",
        "name": "passive",
        "surface": ["ᠭᠳᠠ", "ᠭᠳᠡ", "ᠳᠠ", "ᠳᠡ", "ᠲᠠ", "ᠲᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "passive",
    },
    {
        "id": "RECIP",
        "name": "reciprocal",
        "surface": ["ᠯᠴᠠ", "ᠯᠴᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "reciprocal",
    },
    {
        "id": "COOP",
        "name": "cooperative",
        "surface": ["ᠯᠳᠤ", "ᠯᠳᠦ", "ᠯᠳᠠ", "ᠯᠳᠡ"],
        "type": "voice",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "cooperative",
    },
]


TENSE_SUFFIXES = [
    {
        "id": "FIN_PAST_BA",
        "name": "finite-past",
        "surface": ["ᠪᠠ", "ᠪᠡ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PAST_LUGA",
        "name": "finite-past",
        "surface": ["ᠯᠤᠭ᠎ᠠ", "ᠯᠦᠭᠡ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PRES_MUI",
        "name": "finite-present",
        "surface": ["ᠮᠤᠢ", "ᠮᠦᠢ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_PRES_NAM",
        "name": "finite-present",
        "surface": ["ᠨᠠᠮ", "ᠨᠡᠮ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite",
    },
    {
        "id": "FIN_FUT_QU",
        "name": "finite-future",
        "surface": ["ᠬᠤ", "ᠬᠦ"],
        "type": "tense",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "finite/participle",
    },
]


MOOD_SUFFIXES = [
    {
        "id": "IMP_2",
        "name": "imperative-2",
        "surface": ["ᠠ", "ᠡ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "imperative",
    },
    {
        "id": "IMP_POLITE",
        "name": "imperative-polite",
        "surface": ["ᠭᠠᠷᠠᠢ", "ᠭᠡᠷᠡᠢ", "ᠠᠷᠠᠢ", "ᠡᠷᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "polite imperative",
    },
    {
        "id": "IMP_HON",
        "name": "imperative-honorific",
        "surface": ["ᠭᠲᠤᠨ", "ᠭᠲᠦᠨ", "ᠲᠤᠨ", "ᠲᠦᠨ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "honorific imperative",
    },
    {
        "id": "OPT_1",
        "name": "optative-1",
        "surface": ["ᠶ᠎ᠠ", "ᠶ᠎ᠡ", "ᠰᠤᠭᠠᠢ", "ᠰᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "let me/us",
    },
    {
        "id": "VOLITIVE",
        "name": "volitive",
        "surface": ["ᠰᠤᠭᠠᠢ", "ᠰᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "volitive",
    },
    {
        "id": "PRECATIVE",
        "name": "precative",
        "surface": ["ᠲᠤᠭᠠᠢ", "ᠲᠦᠭᠡᠢ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "may",
    },
    {
        "id": "PROHIBITIVE",
        "name": "prohibitive",
        "surface": ["ᠪᠤᠤ"],
        "type": "mood",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "negative imperative marker",
    },
]


ASPECT_SUFFIXES = [
    {
        "id": "PFV_GAD",
        "name": "perfective",
        "surface": ["ᠭᠠᠳ", "ᠭᠡᠳ", "ᠠᠳ", "ᠡᠳ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "perfective",
    },
    {
        "id": "HAB_DAG",
        "name": "habitual",
        "surface": ["ᠳᠠᠭ", "ᠳᠡᠭ", "ᠲᠠᠭ", "ᠲᠡᠭ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "habitual",
    },
    {
        "id": "PROG_JU",
        "name": "progressive-connective",
        "surface": ["ᠵᠤ", "ᠵᠦ", "ᠴᠤ", "ᠴᠦ"],
        "type": "aspect",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "connective",
    },
]


CONVERB_SUFFIXES = [
    {
        "id": "CVB_COORD_JU",
        "name": "converb-coordinate",
        "surface": ["ᠵᠤ", "ᠵᠦ", "ᠴᠤ", "ᠴᠦ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "and",
        "nnbsp_surfaces": ["ᠴᠤ", "ᠴᠦ"],
    },
    {
        "id": "CVB_SIMUL_N",
        "name": "converb-simultaneous",
        "surface": ["ᠨ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "while",
    },
    {
        "id": "CVB_COND_BAL",
        "name": "converb-conditional",
        "surface": ["ᠪᠠᠯ", "ᠪᠡᠯ", "ᠪᠤᠯ", "ᠪᠦᠯ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "if",
    },
    {
        "id": "CVB_PFV_GAD",
        "name": "converb-perfective",
        "surface": ["ᠭᠠᠳ", "ᠭᠡᠳ", "ᠠᠳ", "ᠡᠳ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "after",
    },
    {
        "id": "CVB_LIMIT_TALA",
        "name": "converb-limitative",
        "surface": ["ᠲᠠᠯ᠎ᠠ", "ᠲᠡᠯ᠎ᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "until",
    },
    {
        "id": "CVB_CONC_BACHU",
        "name": "converb-concessive",
        "surface": ["ᠪᠠᠴᠤ", "ᠪᠡᠴᠦ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "although",
    },
    {
        "id": "CVB_PURP_RA",
        "name": "converb-purposive",
        "surface": ["ᠷᠠ", "ᠷᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "in order to",
    },
    {
        "id": "CVB_PREP_MAGCA",
        "name": "converb-immediate",
        "surface": ["ᠮᠠᠭᠴᠠ", "ᠮᠡᠭᠴᠡ"],
        "type": "converb",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "as soon as",
    },
]


PARTICIPLE_SUFFIXES = [
    {
        "id": "PTCP_PAST_GSAN",
        "name": "participle-past",
        "surface": ["ᠭᠰᠠᠨ", "ᠭᠰᠡᠨ", "ᠰᠠᠨ", "ᠰᠡᠨ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "past",
    },
    {
        "id": "PTCP_FUT_QU",
        "name": "participle-future",
        "surface": ["ᠬᠤ", "ᠬᠦ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "future",
    },
    {
        "id": "PTCP_HAB_DAG",
        "name": "participle-habitual",
        "surface": ["ᠳᠠᠭ", "ᠳᠡᠭ", "ᠲᠠᠭ", "ᠲᠡᠭ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "habitual",
    },
    {
        "id": "PTCP_AGENT_GCHI",
        "name": "participle-agentive",
        "surface": ["ᠭᠴᠢ", "ᠴᠢ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 1,
        "notes": "agentive",
    },
    {
        "id": "PTCP_PRES_A",
        "name": "participle-present",
        "surface": ["ᠠ", "ᠡ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "present",
    },
    {
        "id": "PTCP_POT_MAR",
        "name": "participle-potential",
        "surface": ["ᠮᠠᠷ", "ᠮᠡᠷ"],
        "type": "participle",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 1,
        "notes": "potential",
    },
]


NEGATION_SUFFIXES = [
    {
        "id": "NEG_UGEI",
        "name": "negative",
        "surface": ["ᠦᠭᠡᠢ", "ᠤᠭᠠᠢ"],
        "type": "negation",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 2,
        "notes": "not",
    },
    {
        "id": "NEG_ES",
        "name": "negative-past",
        "surface": ["ᠡᠰᠡ", "ᠡᠰ"],
        "type": "negation",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 2,
        "notes": "negative auxiliary",
    },
]


DERIVATIONAL_SUFFIXES = [
    {
        "id": "DER_N_ADJ_TAI",
        "name": "denominal-adjective",
        "surface": ["ᠲᠠᠢ", "ᠲᠡᠢ", "ᠲᠣᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "with",
    },
    {
        "id": "DER_N_ADJ_UGEI",
        "name": "denominal-negative-adjective",
        "surface": ["ᠦᠭᠡᠢ", "ᠤᠭᠠᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "without",
    },
    {
        "id": "DER_N_V_LA",
        "name": "denominal-verb",
        "surface": ["ᠯᠠ", "ᠯᠡ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "verbalizer",
    },
    {
        "id": "DER_N_V_JI",
        "name": "denominal-verb",
        "surface": ["ᠵᠢ", "ᠴᠢ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "none",
        "order": 0,
        "notes": "verbalizer",
    },
    {
        "id": "DER_V_N_L",
        "name": "deverbal-noun",
        "surface": ["ᠯ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "none",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_V_N_GA",
        "name": "deverbal-noun",
        "surface": ["ᠭᠠ", "ᠭᠡ", "ᠠ", "ᠡ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_V_N_LGA",
        "name": "deverbal-noun",
        "surface": ["ᠯᠭ᠎ᠠ", "ᠯᠭᠡ"],
        "type": "derivational",
        "attaches_to": "verb",
        "harmony": "all_variants",
        "order": 0,
        "notes": "nominalizer",
    },
    {
        "id": "DER_AGENT_CHI",
        "name": "agentive",
        "surface": ["ᠴᠢ", "ᠭᠴᠢ"],
        "type": "derivational",
        "attaches_to": "any",
        "harmony": "none",
        "order": 0,
        "notes": "agent",
    },
    {
        "id": "DER_ABSTRACT_LIG",
        "name": "abstract",
        "surface": ["ᠯᠢᠭ"],
        "type": "derivational",
        "attaches_to": "any",
        "harmony": "none",
        "order": 0,
        "notes": "abstract",
    },
    {
        "id": "DER_QUALITY_TU",
        "name": "quality",
        "surface": ["ᠲᠤ", "ᠲᠦ", "ᠳᠤ", "ᠳᠦ"],
        "type": "derivational",
        "attaches_to": "noun",
        "harmony": "all_variants",
        "order": 0,
        "notes": "having quality",
    },
    {
        "id": "DER_DIM_QAN",
        "name": "diminutive",
        "surface": ["ᠬᠠᠨ", "ᠬᠡᠨ", "ᠬᠤᠨ", "ᠬᠦᠨ"],
        "type": "derivational",
        "attaches_to": "adj",
        "harmony": "all_variants",
        "order": 0,
        "notes": "diminutive",
    },
]


PARTICLE_SUFFIXES = [
    {
        "id": "Q_UU",
        "name": "interrogative-particle",
        "surface": ["ᠤᠤ", "ᠦᠦ"],
        "type": "particle",
        "attaches_to": "clause",
        "harmony": "all_variants",
        "order": 4,
        "notes": "yes/no question particle",
        "nnbsp": True,
    },
]


LEGACY_MONGOL_CODE_SUFFIX_SURFACES = frozenset(
    [
        "ᠶᠢᠨ",
        "ᠤᠨ",
        "ᠦᠨ",
        "ᠤ",
        "ᠦ",
        "ᠢ",
        "ᠶᠢ",
        "ᠳᠤ",
        "ᠳᠦ",
        "ᠲᠤ",
        "ᠲᠦ",
        "ᠳᠤᠷ",
        "ᠳᠦᠷ",
        "ᠲᠤᠷ",
        "ᠲᠦᠷ",
        "ᠳᠠᠬᠢ",
        "ᠳᠡᠬᠢ",
        "ᠲᠠᠬᠢ",
        "ᠲᠡᠬᠢ",
        "ᠠᠴᠠ",
        "ᠡᠴᠡ",
        "ᠪᠠᠷ",
        "ᠪᠡᠷ",
        "ᠢᠶᠠᠷ",
        "ᠢᠶᠡᠷ",
        "ᠲᠠᠢ",
        "ᠲᠡᠢ",
        "ᠯᠤᠭ᠎ᠠ",
        "ᠯᠦᠭᠡ",
        "ᠪᠠᠨ",
        "ᠪᠡᠨ",
        "ᠢᠶᠠᠨ",
        "ᠢᠶᠡᠨ",
        "ᠶᠤᠭᠠᠨ",
        "ᠶᠦᠭᠡᠨ",
        "ᠳᠠᠭᠠᠨ",
        "ᠳᠡᠭᠡᠨ",
        "ᠲᠠᠭᠠᠨ",
        "ᠲᠡᠭᠡᠨ",
        "ᠠᠴᠠᠭᠠᠨ",
        "ᠡᠴᠡᠭᠡᠨ",
        "ᠲᠠᠢᠭᠠᠨ",
        "ᠲᠡᠢᠭᠡᠨ",
        "ᠤᠳ",
        "ᠦᠳ",
        "ᠨᠤᠭᠤᠳ",
        "ᠨᠦᠭᠦᠳ",
        "ᠨᠠᠷ",
        "ᠨᠡᠷ",
        "ᠤᠤ",
        "ᠦᠦ",
        "ᠳᠠ",
        "ᠳᠡ",
        "ᠴᠤ",
        "ᠴᠦ",
    ]
)


CORE_TRADITIONAL_MONGOLIAN_SUFFIXES = (
    CASE_SUFFIXES
    + PLURAL_SUFFIXES
    + POSSESSIVE_SUFFIXES
    + VOICE_SUFFIXES
    + TENSE_SUFFIXES
    + MOOD_SUFFIXES
    + ASPECT_SUFFIXES
    + CONVERB_SUFFIXES
    + PARTICIPLE_SUFFIXES
    + NEGATION_SUFFIXES
    + DERIVATIONAL_SUFFIXES
    + PARTICLE_SUFFIXES
)

ALL_SUFFIXES = CORE_TRADITIONAL_MONGOLIAN_SUFFIXES

ALL_SUFFIXES_BY_ORDER = sorted(
    ALL_SUFFIXES,
    key=lambda item: (
        -item.get("order", 0),
        -max(len(strip_all(surface)) for surface in item.get("surface", [""])),
        item.get("id", ""),
    ),
)


def with_nnbsp(suffix):
    return NNBSP + suffix if suffix else suffix


def allows_nnbsp(item, surface):
    if item.get("nnbsp", False):
        return True
    return surface in item.get("nnbsp_surfaces", ())


def iter_surfaces(include_nnbsp=False, include_empty=False):
    for item in ALL_SUFFIXES_BY_ORDER:
        for surface in item["surface"]:
            if not surface and not include_empty:
                continue
            if include_nnbsp and allows_nnbsp(item, surface):
                yield item, with_nnbsp(surface)
            else:
                yield item, surface


def duplicate_surfaces():
    surface_to_ids = {}
    for item in ALL_SUFFIXES:
        for surface in item["surface"]:
            surface_to_ids.setdefault(surface, []).append(item["id"])
    return {
        surface: ids
        for surface, ids in surface_to_ids.items()
        if surface and len(ids) > 1
    }


def validate_suffix_inventory():
    ids = [item["id"] for item in ALL_SUFFIXES]
    duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate suffix ids: {duplicate_ids}")

    for item in ALL_SUFFIXES:
        if not item.get("surface"):
            raise ValueError(f"Suffix {item['id']} has no surfaces")
        if any(not isinstance(surface, str) for surface in item["surface"]):
            raise TypeError(f"Suffix {item['id']} has a non-string surface")
        if any(surface == "" for surface in item["surface"]):
            raise ValueError(f"Suffix {item['id']} has an empty surface")

    surfaces = {surface for item in ALL_SUFFIXES for surface in item["surface"]}
    missing_legacy = sorted(LEGACY_MONGOL_CODE_SUFFIX_SURFACES - surfaces)
    if missing_legacy:
        raise ValueError(
            f"Missing legacy mongol_code suffix surfaces: {missing_legacy}"
        )

    return True


validate_suffix_inventory()
