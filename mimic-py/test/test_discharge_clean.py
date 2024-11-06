import pandas as pd
from pandas.testing import assert_frame_equal
from lib.discharge_clean import get_process_columns, find_text_between_word_pairs, create_word_pair_tuple, \
    get_process_columns_split

col_names = ["Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Sex", "Service", "Allergies",
             "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
             "Past Medical History", "Social History", "Family History", "Physical Exam", "Discharge",
             "Pertinent Results", "CXR", "U/S", "Brief Hospital Course", "Medications on Admission",
             "Discharge Medications", "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition",
             "Discharge Instructions", "Followup Instructions"]


def test_process_columns():
    # Arrange
    expected_val = pd.DataFrame({'Chief Complaint': ["Worsening ABD distension and pain",
                                                     "abdominal fullness and discomfort",
                                                     "altered mental status"]})
    df = pd.read_csv("fixture/100_discharge.csv")

    # Act
    df = df.loc[:2].apply(get_process_columns(col_names), axis=1)
    returned_val = df[['Chief Complaint']]

    # Assert
    assert_frame_equal(returned_val, expected_val)


def test_split_between_text_pairs():
    # Arrange
    text = """
    This is some sample text. StartA Here is the first chunk EndA.
    Another section StartB This is the second chunk EndB. 
    Yet more text StartA Another chunk for the first pair EndA.
    And finally StartC Last distinct chunk EndC.
    """

    # Start and end word pairs
    limiting_pairs = [
        ("StartA", "EndA"),
        ("StartB", "EndB"),
        ("StartC", "EndC")
    ]

    expected_val = [('StartA', 'EndA', 'Here is the first chunk'),
                    ('StartA', 'EndA', 'Another chunk for the first pair'),
                    ('StartB', 'EndB', 'This is the second chunk'), ('StartC', 'EndC', 'Last distinct chunk')]

    # Act
    # Get chunks
    # Loop through each start-end word pair
    chunks = find_text_between_word_pairs(text, limiting_pairs)

    # Assert
    assert all(x == y for x, y in zip(expected_val, chunks))


def test_create_word_pair_tuple():
    # Arrange
    limiting_words = ["StartA", "StartB", "EndB", "StartC", "EndC"]
    expected_output = [
        ("StartA", "StartB"),
        ("StartB", "EndB"),
        ("EndB", "StartC"),
        ("StartC", "EndC"),
    ]

    # Act
    pairs = create_word_pair_tuple(limiting_words)

    # Assert
    assert all(x == y for x, y in zip(expected_output, pairs))


def test_get_process_columns_split():
    # Arrange
    limiting_words = ["Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
                      "Past Medical History"]
    expected_output = ["Worsening ABD distension and pain", "abdominal fullness and discomfort",
                       "altered mental status"]
    df = pd.read_csv("fixture/100_discharge.csv").loc[:2]

    # Act
    apply_func = get_process_columns_split(limiting_words)
    df = df.apply(apply_func, axis=1)

    # Assert
    assert all(x == y for x, y in zip(expected_output, df["Chief Complaint"].tolist()))
