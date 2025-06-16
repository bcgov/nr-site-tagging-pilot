import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Hugging Face model globals
hf_tokenizer = None
hf_model = None
label_encoder = None  # Optional, if using label indices

# Keyword map for regex classification
DOCUMENT_TYPES = {
    "AiP": ["approval in principle", "aip"],
    "CoC": ["certificate of compliance", "coc"],
    "FDet": ["final determination", "fdet"],
    "PDet": ["preliminary determination", "pdet"],
    "PSI": ["preliminary site investigation", "stage 1 psi", "stage 2 psi", "psi"],
    "DSI": ["detailed site investigation", "dsi"],
    "COV": ["covenant", "cov"],
    "CORR": ["correspondence", "letter", "email", "corr"],
    "RPT": ["remediation", "report", "summary", "investigation"],
}


DOCUMENT_CLASS_NAMES = ['AIP', 'COA', 'COC', 'CORR', 'COV', 'CSSA', 'DSI', 'FDET', 'IMG',
                        'MAP', 'NIR', 'NIRI', 'NOM', 'OTHERS', 'PDET', 'PSI', 'RA', 'RPT',
                        'SP', 'SSI', 'Site Registry', 'TITLE', 'TMEMO']


def load_huggingface_model(model_name, device):
    """
    Loads a Hugging Face transformer model and tokenizer for document classification.

    Parameters:
    ----------
    model_name : str
        Name or path of the pretrained model to load.
    device : torch.device
        The device (CPU/GPU) to move the model to.

    Side Effects:
    -------------
    Sets global variables:
    - hf_tokenizer: The loaded tokenizer.
    - hf_model: The loaded model in evaluation mode.

    Returns
    -------------
    None
    """
    global hf_tokenizer, hf_model
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_model.to(device)
    hf_model.eval()


def classify_document(file_path, device, metadata=None, mode="regex"):
    """
    Classifies a document either using a regex-based method or a transformer-based ML model.

    Parameters:
        file_path (Path): Path to the document file.
        metadata (dict): Optional dictionary containing extracted metadata (used by ML).
        mode (str): Classification mode: "regex" (default) or "ml".

    Returns:
        str: Predicted document type label (e.g., "REPORT", "PSI").
    """
    if mode == "ml":
        try:
            return classify_with_ml(device, metadata)
        except Exception as e:
            print(
                f"[ML fallback] Classification failed: {e}. Falling back to regex.")
    return classify_with_regex(file_path)


def classify_with_regex(file_path):
    """
    Classifies a document by checking for keyword matches in the filename.

    Parameters:
        file_path (Path): Path to the document file.

    Returns:
        str: Document type label if matched; otherwise, "REPORT" as fallback.
    """
    filename = file_path.name.lower()

    for doc_type, keywords in DOCUMENT_TYPES.items():
        for kw in keywords:
            if kw in filename:
                return doc_type
    return "CORR"  # default fallback if nothing matches


def classify_with_ml(device, metadata=None):
    """
    Classifies a document using a fine-tuned Hugging Face transformer model.

    Parameters:
        file_path (Path): Path to the document file (not used in this method).
        metadata (dict): Dictionary containing at least a "title" field for classification.

    Returns:
        str: Predicted document type label (from label encoder or index).

    Raises:
        ValueError: If the Hugging Face model/tokenizer is not loaded.
    """
    if hf_tokenizer is None or hf_model is None:
        raise ValueError(
            "Hugging Face model not loaded. Call load_huggingface_model() first.")

    title = metadata.get("title", "").strip()

    inputs = hf_tokenizer(title, return_tensors="pt",
                          truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = hf_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return DOCUMENT_CLASS_NAMES[predicted_class]
