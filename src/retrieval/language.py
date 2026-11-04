# Guesses whether a text is Finnish or English by counting common small words.


import re

FINNISH = set("""ja on ei että se ne oli ovat olla kanssa mutta tai kun jos myös sekä vain
voi kuin mitä joka jotka tämä nämä siitä sen hän me te he minä sinä tässä siinä mukaan
ennen jälkeen kautta noin osalta kiitos terveisin hei moi yhteistyö toimitus laite
mittaus näyte""".split())

ENGLISH = set("""the and is of to in with for that this are be it on as by from at or an
was were have has will can not we you your our please regards thanks dear device
measurement sample""".split())

_WORD = re.compile(r"[a-zäöå]+")


# Guess if a text is Finnish, English, mixed, or mostly numbers.
def detect_language(text):
    words = _WORD.findall((text or "").lower())
    fi = sum(w in FINNISH for w in words)
    en = sum(w in ENGLISH for w in words)
    if fi >= 2 and fi > en * 1.5:
        return "fi"
    if en >= 2 and en > fi * 1.5:
        return "en"
    if fi + en >= 2:
        return "mixed"
    return "little text"
