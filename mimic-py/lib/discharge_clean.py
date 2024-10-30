import pandas as pd
import re


def history_of_present_illness(notes_df):
    in_text = notes_df.text
    in_text = in_text.replace("\n", " ")
    in_text = in_text.replace("\r", " ")

    replace_with = ''
    everything_before = r".*History of Present Illness:"
    text = re.sub(everything_before, replace_with, in_text)

    everything_after = r"Past Medical History:.*"
    text = re.sub(everything_after, replace_with, text)

    notes_df.history_of_present_illness = text.strip()
    return notes_df


def chief_complaint(notes_df):
    in_text = notes_df.text
    in_text = in_text.replace("\n", " ")
    in_text = in_text.replace("\r", " ")

    replace_with = ''
    everything_before = r".*Chief Complaint:"
    everything_after = r"Major Surgical or Invasive Procedure:.*"

    regexp_first = re.compile(everything_before)
    regexp_last = re.compile(everything_after)

    ## Make sure chief complaints are present.
    text = "chief complaint absent"
    if regexp_first.search(in_text) and regexp_last.search(in_text):
        text = re.sub(everything_before, replace_with, in_text)
        text = re.sub(everything_after, replace_with, text)

    notes_df.chief_complaint = text.strip()
    return notes_df


def discharge_diagnosis(notes_df):
    in_text = notes_df.text
    in_text = in_text.replace("\n", " ")
    in_text = in_text.replace("\r", " ")

    replace_with = ''
    everything_before = r".*Discharge Diagnosis:"
    text = re.sub(everything_before, replace_with, in_text)

    everything_after = r"Discharge Condition:.*"
    text = re.sub(everything_after, replace_with, text)

    everything_before = r".*___ Diagnosis:"
    text = re.sub(everything_before, replace_with, text)

    notes_df.discharge_diagnosis = text.strip()
    return notes_df


def do_processing(chunk_df):
    columns = ['note_id', 'subject_id', 'note_type', 'note_seq', 'history_of_present_illness', 'chief_complaint',
               'discharge_diagnosis']
    chunk_df.loc[:, 'history_of_present_illness'] = ''
    chunk_df.loc[:, 'chief_complaint'] = ''
    chunk_df.loc[:, 'discharge_diagnosis'] = ''
    df_after = chunk_df.apply(history_of_present_illness, axis=1)
    df_after = df_after.apply(chief_complaint, axis=1)
    df_after = df_after.apply(discharge_diagnosis, axis=1)
    df_after = df_after.loc[:, columns]
    return df_after


def clean_discharge_in_chunks(args, discharge_df):
    print("\n=============================================")
    print("Cleaning discharge chunks")

    pd.options.mode.copy_on_write = True
    all_data_frames = []
    chunk_no = 1

    for chunk in discharge_df:
        chunk = do_processing(chunk)
        print(f"Processed chunk no: {chunk_no}")
        all_data_frames.append(chunk)
        chunk_no += 1
        # If testing break after 1 iteration
        if not args.full_run:
            break

    return pd.concat(all_data_frames)
