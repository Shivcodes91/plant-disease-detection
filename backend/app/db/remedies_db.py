# =============================================================
# backend/app/db/remedies_db.py
#
# Remedies database module.
# Loads the remedies JSON and provides query functions.
#
# This is what the API calls when a disease is detected:
# get_remedy("Tomato_Early_blight") → full treatment info
# =============================================================

import json
from pathlib import Path
from typing import Dict, Optional, List

# Path to our remedies JSON file
REMEDIES_PATH = Path('data/remedies.json')


def load_remedies() -> Dict:
    """Load the entire remedies database from JSON."""
    if not REMEDIES_PATH.exists():
        raise FileNotFoundError(
            f"Remedies file not found at {REMEDIES_PATH}\n"
            f"Make sure data/remedies.json exists!"
        )
    with open(REMEDIES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_remedy(class_name: str) -> Optional[Dict]:
    """
    Get full remedy information for a disease class.

    Args:
        class_name: exact class name from model
                    e.g. "Tomato_Early_blight"

    Returns:
        Dictionary with full remedy info, or None if not found
    """
    db = load_remedies()
    diseases = db.get('diseases', {})

    # Direct lookup
    if class_name in diseases:
        remedy = diseases[class_name].copy()
        remedy['found'] = True
        return remedy

    # Try case-insensitive match
    for key, value in diseases.items():
        if key.lower() == class_name.lower():
            remedy = value.copy()
            remedy['found'] = True
            return remedy

    return {'found': False, 'class_name': class_name}


def get_all_diseases() -> List[Dict]:
    """
    Returns summary list of all diseases.
    Used for the /diseases API endpoint.
    """
    db = load_remedies()
    diseases = []

    for class_name, info in db['diseases'].items():
        diseases.append({
            'class_name': class_name,
            'display_name': info['display_name'],
            'plant': info['plant'],
            'severity': info['severity'],
            'is_healthy': info['is_healthy'],
            'pathogen_type': info.get('pathogen_type'),
        })

    return diseases


def get_quick_summary(class_name: str) -> Dict:
    """
    Returns a quick summary for the prediction result card.
    Used directly in the /predict API response.

    Args:
        class_name: disease class name from model

    Returns:
        Compact summary dictionary
    """
    remedy = get_remedy(class_name)

    if not remedy or not remedy.get('found'):
        return {
            'found': False,
            'display_name': class_name,
            'message': 'Remedy information not available'
        }

    # Get first immediate action
    immediate = remedy.get('treatments', {}).get(
        'immediate_action', ['Consult an agronomist']
    )

    # Get first chemical treatment if available
    chemicals = remedy.get('treatments', {}).get(
        'chemical_treatments', []
    )
    first_chemical = chemicals[0] if chemicals else None

    return {
        'found': True,
        'display_name': remedy['display_name'],
        'plant': remedy['plant'],
        'severity': remedy['severity'],
        'is_healthy': remedy['is_healthy'],
        'overview': remedy['overview'],
        'pathogen_type': remedy.get('pathogen_type'),
        'pathogen_name': remedy.get('pathogen_name'),
        'immediate_action': immediate[0] if immediate else None,
        'first_treatment': first_chemical,
        'prevention_count': len(remedy.get('prevention', [])),
        'has_organic_options': len(
            remedy.get('treatments', {}).get('organic_treatments', [])
        ) > 0
    }