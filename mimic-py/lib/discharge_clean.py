import pandas as pd
import re
import time


# def chief_complaint(notes_df):
#     in_text = notes_df.text
#     in_text = in_text.replace("\n", " ")
#     in_text = in_text.replace("\r", " ")
#
#     replace_with = ''
#     everything_before = r".*Chief Complaint:"
#     everything_after = r"Major Surgical or Invasive Procedure:.*"
#
#     regexp_first = re.compile(everything_before)
#     regexp_last = re.compile(everything_after)
#
#     ## Make sure chief complaints are present.
#     text = "chief complaint absent"
#     if regexp_first.search(in_text) and regexp_last.search(in_text):
#         text = re.sub(everything_before, replace_with, in_text)
#         text = re.sub(everything_after, replace_with, text)
#
#     notes_df.chief_complaint = text.strip()
#     return notes_df


# def discharge_diagnosis(notes_df):
#     in_text = notes_df.text
#     in_text = in_text.replace("\n", " ")
#     in_text = in_text.replace("\r", " ")
#
#     replace_with = ''
#     everything_before = r".*Discharge Diagnosis:"
#     text = re.sub(everything_before, replace_with, in_text)
#
#     everything_after = r"Discharge Condition:.*"
#     text = re.sub(everything_after, replace_with, text)
#
#     everything_before = r".*___ Diagnosis:"
#     text = re.sub(everything_before, replace_with, text)
#
#     notes_df.discharge_diagnosis = text.strip()
#     return notes_df


def split_between_text_pairs(text, start_word, end_word):
    # Store the results for each pair of start and end words
    all_chunks = []

    # Create a regex pattern to capture text between each start and end word pair
    start_word_to_search = start_word + ":"
    pattern = re.compile(f"{re.escape(start_word_to_search)}(.*?){re.escape(end_word)}", re.DOTALL)
    # Find all matches for the current pair and add them to the list
    chunks = pattern.findall(text)
    # Append the chunks to the main list with a label for each pair
    all_chunks.extend([(start_word, end_word, chunk.strip()) for chunk in chunks])

    return all_chunks


def find_text_between_word_pairs(text, limiting_pairs):
    chunks = []
    for start_word, end_word in limiting_pairs:
        chunks.extend(split_between_text_pairs(text, start_word, end_word))
    return chunks


def create_word_pair_tuple(limiting_words):
    pairs = []
    for i in range(len(limiting_words) - 1):
        pairs.append((limiting_words[i], limiting_words[i + 1]))
    return pairs


def get_process_columns_split(col_names):
    limiting_pairs = create_word_pair_tuple(limiting_words=col_names)

    def process_columns_split(df):
        text = df['text']
        chunk = find_text_between_word_pairs(text=text, limiting_pairs=limiting_pairs)
        for first_word, second_word, found_text in chunk:
            df[first_word] = found_text.strip()
        return df

    return process_columns_split


# def get_process_columns(col_names):
#     def process_columns(df):
#         for i in range(len(col_names) - 1):
#             text = df['text']
#             text = text.replace("\n", " ")
#             text = text.replace("\r", " ")
#             replace_with = ''
#             everything_before = fr".*{col_names[i]}:"
#             text = re.sub(everything_before, replace_with, text)
#
#             everything_after = fr"{col_names[i + 1]}:.*"
#             text = re.sub(everything_after, replace_with, text)
#
#             df[col_names[i]] = text.strip()
#         return df
#
#     return process_columns


def clean_discharge_in_chunks(args, discharge_df, col_names):
    print("\n=============================================")
    print("Cleaning discharge chunks")

    pd.options.mode.copy_on_write = True
    all_data_frames = []
    chunk_no = 1

    for chunk in discharge_df:
        start_time = time.time()
        chunk = chunk.apply(get_process_columns_split(col_names), axis=1)

        print(f"Processed chunk no: {chunk_no}")
        print("Seconds taken: %s" % (time.time() - start_time))
        all_data_frames.append(chunk)
        chunk_no += 1
        # If testing break after 1 iteration
        if not args.full_run:
            break

    return pd.concat(all_data_frames)
