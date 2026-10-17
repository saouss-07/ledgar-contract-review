import pandas as pd

from ledgar.metadata import (METADATA_COLUMNS, add_metadata, contract_type, find_durations, find_jurisdiction,
                             gender, jurisdiction_group, party_type)


def test_jurisdiction_us_states():
    assert find_jurisdiction("governed by the laws of the State of New York, without regard to") == "New York"
    assert find_jurisdiction("THE LAW OF THE STATE OF DELAWARE") == "Delaware"
    assert find_jurisdiction("subject to Delaware law") == "Delaware"
    assert find_jurisdiction("in accordance with the Delaware General Corporation Law") == "Delaware"
    assert find_jurisdiction("the laws of the Commonwealth of Massachusetts") == "Massachusetts"
    # the longest name wins
    assert find_jurisdiction("the laws of the State of West Virginia") == "West Virginia"


def test_jurisdiction_foreign_and_none():
    assert find_jurisdiction("governed by the laws of England and Wales") == "England and Wales"
    assert find_jurisdiction("the courts of Ontario shall have jurisdiction") == "Ontario"
    assert find_jurisdiction("This Agreement may be executed in counterparts.") is None
    # an address is not a governing law
    assert find_jurisdiction("notices shall be sent to 10 Main Street, New York, NY") is None


def test_jurisdiction_group():
    assert jurisdiction_group("Delaware") == "Delaware"
    assert jurisdiction_group("Ohio") == "autre État US"
    assert jurisdiction_group("England and Wales") == "hors US"
    assert jurisdiction_group(None) == "aucune"


def test_durations():
    assert find_durations("upon thirty (30) days prior written notice") == [("30", "days")]
    assert find_durations("within 45 calendar days") == [("45", "days")]
    assert find_durations("for a period of six months") == [("six", "months")]
    assert find_durations("Section 30 of the Agreement") == []


def test_party_and_contract_type():
    assert party_type("The Executive shall devote his full business time") == "personne physique"
    assert party_type("each Lender and the Administrative Agent") == "institution financière"
    assert party_type("The Company shall indemnify") == "société"
    assert party_type("This Agreement may be executed in counterparts.") == "non précisé"
    assert contract_type("The Borrower shall repay the Loan to each Lender") == "financement"
    assert contract_type("The Tenant shall use the Premises") == "bail"
    assert contract_type("This Agreement may be executed in counterparts.") == "inconnu"


def test_gender_does_not_match_inside_words():
    assert gender("the Executive and his heirs") == "masculin"
    assert gender("she shall be entitled") == "féminin"
    assert gender("his or her death") == "mixte"
    assert gender("hereunder, therein and thereof") == "aucun"


def test_add_metadata_columns():
    df = pd.DataFrame({"text": ["THE LAWS OF THE STATE OF TEXAS SHALL GOVERN.", "Fees of $5,000 are due within 10 days."]})
    out = add_metadata(df)
    assert set(METADATA_COLUMNS) <= set(out.columns)
    assert out.loc[0, "jurisdiction"] == "Texas" and bool(out.loc[0, "all_caps"])
    assert bool(out.loc[1, "has_amount"]) and bool(out.loc[1, "has_duration"])
    assert list(out["length_group"]) == ["court", "court"]
    # a clause without any place must land in "aucune" through the DataFrame path too (NaN, not None)
    assert out.loc[1, "jurisdiction_group"] == "aucune"
