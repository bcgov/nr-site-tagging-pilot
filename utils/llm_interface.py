import ollama
import re
from difflib import SequenceMatcher


def load_prompt_template(path, doc_text: str) -> str:
    """
    Loads a prompt template from file and injects the document text into the {{DOCUMENT_TEXT}} placeholder.

    Parameters:
    ----------
    path : str or Path
        File path to the prompt template.
    doc_text : str
        The document text to inject into the template (truncated to 3000 characters).

    Returns:
    -------
    str
        Prompt string with the document text inserted.
    """
    with open(path, "r") as file:
        template = file.read()
    return template.replace("{{DOCUMENT_TEXT}}", doc_text.strip()[:3000])


def query_llm(prompt, model="llama2", system_prompt=None):
    """
    Queries an LLM via the Ollama API to extract a metadata dictionary.

    Parameters:
    ----------
    prompt : str
        The prompt to send to the LLM.
    model : str
        The LLM model name to query (default is "llama2").
    system_prompt : str, optional
        An optional system-level prompt to prepend to the conversation.

    Returns:
    -------
    dict
        Metadata dictionary with keys like 'site_id', 'title', etc. Defaults to 'none' values on failure.
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    try:
        response = ollama.chat(model=model, messages=messages)
        raw = response['message']['content'].strip()
        metadata_dict = eval(raw)

    except Exception as e:
        print(e)
        metadata_dict = {
            "site_id": "none",
            "title": "none",
            "receiver": "none",
            "sender": "none",
            "address": "none",
            "readable": "no"
        }

    return metadata_dict


def llm_single_field_query(prompt, model="llama2", system_prompt=None) -> str:
    """
    Queries the LLM for a single metadata field (e.g., title or site_id).

    Parameters:
    ----------
    prompt : str
        Prompt specific to the single field being extracted.
    model : str
        The LLM model name to query.
    system_prompt : str, optional
        Optional system prompt.

    Returns:
    -------
    str
        The extracted value as a string.
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    response = ollama.chat(model=model, messages=messages)
    raw = response['message']['content'].strip()

    return raw


def all_words_in_text(field, text):
    """
    Checks whether all words in the field are present in the source document text.

    Useful for verifying if the LLM-generated field is hallucinated or not.

    Parameters:
    ----------
    field : str
        The extracted metadata field.
    text : str
        The OCR-cleaned text from the original document.

    Returns:
    -------
    bool
        True if all words in the field exist in the document text, else False.
    """

    # Remove all punctuation and digits before checking (LLM occasionally adds harmless punctuation)
    field = re.sub(r'[^\w\s]', ' ', field)
    field = re.sub(r'\d+', ' ', field)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # If field is nothing but whitespace, punctuation, or digits, reject it
    if not field.strip():
        return False

    # If any word in field does not occur in actual text, reject it
    for word in field.lower().split():
        if word not in text.lower().split():
            return False
    return True


def field_is_well_formed(field, text, length):
    """
    Determines if a metadata field is valid based on minimum token length and presence in text.

    Parameters:
    ----------
    field : str
        The extracted field value.
    text : str
        OCR-cleaned text from which the field was generated.
    length : int
        Minimum number of words the field must contain.

    Returns:
    -------
    bool
        True if field is valid, else False.
    """

    if len(field.split()) < length and all_words_in_text(field, text):
        return True
    return False


def validate_and_reprompt_field(field_name, length, reprompt_path, metadata_dict, text, filename, flagged_for_review, max_retries=5):
    """
    Validates a specific field in the metadata dictionary. If invalid, re-prompts LLM up to `max_retries` times.

    If the field is still invalid after all retries, the field is flagged for manual review.

    Parameters:
    ----------
    field_name : str
        Key in metadata_dict to validate and possibly re-prompt.
    length : int
        Minimum number of words required for the field to be considered valid.
    reprompt_path : Path
        Path to the prompt template used for re-prompting the field.
    metadata_dict : dict
        Metadata dictionary returned from the initial LLM query.
    text : str
        OCR-cleaned document text for validation and context.
    filename : str
        Name of the document being processed (used for logging).
    flagged_for_review : collections.defaultdict(list)
        Dictionary to collect fields needing manual verification.
    max_retries : int, optional
        Maximum number of re-prompting attempts before giving up (default is 5).

    Returns:
    -------
    None
    """
    field_value = metadata_dict[field_name].strip().lower()
    if field_value != 'none':
        retries = 0
        while (
            metadata_dict[field_name].strip().lower() != 'none' and
            not field_is_well_formed(metadata_dict[field_name], text, length=length) and
            retries < max_retries
        ):
            print(
                f"Retrying {field_name} extraction, attempt {retries + 1}/{max_retries}")
            reprompt = load_prompt_template(reprompt_path, text)
            metadata_dict[field_name] = llm_single_field_query(
                reprompt, model="mistral")
            retries += 1

        if metadata_dict[field_name].strip().lower() != 'none' and not field_is_well_formed(metadata_dict[field_name], text, length=length):
            flagged_for_review[filename].append(field_name)
            print(f"{filename} flagged for manual review: {field_name.upper()}")


def keys_are_well_formed(metadata_dict):
    """
    Validates that `metadata_dict` contains exactly the expected keys.

    This function checks whether the dictionary has all and only the following keys:
    - "site_id"
    - "title"
    - "receiver"
    - "sender"
    - "address"
    - "readable"

    Parameters:
    ----------
    metadata_dict : dict
        Dictionary containing extracted metadata fields from an LLM or parser.

    Returns:
    -------
    bool
        True if all and only expected keys are present; False otherwise.
    """
    if metadata_dict.keys() == {
        "site_id",
        "title",
        "receiver",
        "sender",
        "address",
        "readable"
    }:
        return True
    return False
