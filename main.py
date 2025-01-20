import json

import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
# Initialize the ChromaDB Persistent Client
chroma_client = chromadb.PersistentClient(path="database")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=st.secrets["OpenAI_key"],
    model_name="text-embedding-3-small"
)

#collection = chroma_client.get_collection(name="drugs_repo")
collection = chroma_client.get_or_create_collection(name="drugs_repo",
                                                    metadata={"hnsw:space": "cosine"},
                                                    embedding_function=openai_ef)

open_ai_client = OpenAI(api_key=st.secrets["OpenAI_key"])

find_right_query_prompt = """
**Instructions to Construct the Search Text Query for the Medical Database**

To find the appropriate over-the-counter (OTC) medication in the French medical database based on patient information and symptoms, follow these steps:
### **Step 1: Collect Patient Information and Symptoms**

Gather all relevant details from the patient's information and OTC request:

- **Patient Information:**
  - **Age**
  - **Gender**
  - **Weight**
  - **Allergies**
  - **Current Medications**
  - **Pregnancy/Breastfeeding Status**
  - **Other Information** (e.g., medical history, chronic conditions)

- **OTC Request:**
  - **Primary Symptoms**
  - **Severity and Duration**
  - **Previous Treatments**

---

### **Step 2: Identify Key Factors Affecting Medication Choice**

Analyze the collected information to determine factors that will influence the search:

1. **Primary Symptoms:**
   - Note the main symptoms that need to be treated.

2. **Allergies:**
   - Identify any known drug allergies (e.g., aspirin, ibuprofen).

3. **Current Medications:**
   - Consider possible drug interactions.

4. **Pregnancy/Breastfeeding Status:**
   - Determine if the patient is pregnant or breastfeeding, as this limits medication options.

5. **Age and Weight:**
   - Some medications have age or weight restrictions.

6. **Medical History:**
   - Note any chronic conditions (e.g., asthma, hypertension) that may contraindicate certain medications.

---

### **Step 3: Translate Symptoms and Relevant Terms into French**

Since the medical database is in French, translate the following into French:

- **Primary Symptoms**
- **Medical Conditions**
- **Allergies and Contraindicated Substances**
- **Patient-Specific Factors (e.g., "enceinte" for pregnant)**

*Example:*
- Headache → "maux de tête"
- Fever → "fièvre"
- Allergic to ibuprofen → "allergie à l'ibuprofène"

---

### **Step 4: Formulate Search Keywords**

Based on the translated terms, create a list of keywords:

1. **Symptoms/Indications Keywords:**
   - Symptoms to be treated (e.g., "maux de tête", "fièvre").

2. **Medication Type:**
   - Desired medication category (e.g., "analgésique" for pain reliever, "antipyrétique" for fever reducer).

3. **Exclusions (Allergies/Contraindications):**
   - Substances to avoid (e.g., "sans ibuprofène", "sans aspirine").

4. **Patient-Specific Considerations:**
   - Special conditions (e.g., "femme enceinte", "allaitement", "enfant de moins de 12 ans").

---

### **Step 5: Construct the Search Query Using Boolean Operators**

Combine the keywords into a coherent search query using Boolean operators in French:

- **AND → "ET"**
- **OR → "OU"**
- **NOT → "SAUF"**

*Structure:*
```
[Medication Type] ET [Symptoms] ET [Patient Considerations] SAUF [Exclusions]
```

*Example:*
```
"analgésique ET maux de tête ET femme enceinte SAUF ibuprofène"
```

---

### **Step 6: Refine the Search Query**

1. **Include Synonyms and Related Terms:**
   - Enhance the query with synonyms to ensure comprehensive results.
   - Example: "douleur" (pain), "antalgique" (painkiller).

2. **Use Quotation Marks for Exact Phrases:**
   - Enclose multi-word terms in quotes.
   - Example: "\"maux de tête\""

3. **Adjust for Severity and Duration:**
   - If necessary, include terms related to severity (e.g., "douleur modérée") or duration (e.g., "aiguë", "chronique").

---

### **Step 7: Apply Filters and Advanced Search Options**


1. **Filter by Patient Age or Weight:**
   - Specify if the patient is a child or adult.

2. **Filter by Pregnancy/Breastfeeding:**
   - Choose medications safe for pregnant or breastfeeding women.

3. **Exclude Specific Drug Classes:**
   - If allergic to NSAIDs ("AINS"), exclude them from the search.


---
**Example Application:**

- **Patient:**
  - Adult woman, pregnant.
  - Allergic to ibuprofen.
  - Primary symptom: Headache.

Final Result: **Search Query:**
Return JSON:
  {
  search_query: "analgésique ET maux de tête ET femme enceinte SAUF ibuprofène"
  }
"""

find_right_recommended_drugs = """
To determine which over-the-counter (OTC) medications to recommend or exclude based on the patient's information and the available drug list, follow these steps:

### **Step 1: Apply Filters and Advanced Search Options**
1. **Filter by Patient Age or Weight:**
   - Specify if the patient is a child or adult.

2. **Filter by Pregnancy/Breastfeeding:**
   - Choose medications safe for pregnant or breastfeeding women.

3. **Exclude Specific Drug Classes:**
   - If allergic to NSAIDs ("AINS"), exclude them from the search.


### **Step 3: Evaluate Each Drug in the List**

For each drug in the retrieved list, perform the following evaluations:

1. **Check Active Ingredients:**
   - **Exclude** drugs containing substances the patient is allergic to.
   - Compare the active ingredients with the list of known allergens.

2. **Review Contraindications:**
   - Read the "Contre-indications" section.
   - **Exclude** drugs that are contraindicated for the patient's conditions or status (e.g., pregnancy, asthma).

3. **Assess Drug Interactions:**
   - Review the "Interactions avec d'autres médicaments" section.
   - **Exclude** drugs that have significant interactions with the patient's current medications.

4. **Consider Special Warnings and Precautions:**
   - Examine the "Mises en garde spéciales et précautions d'emploi" section.
   - **Exclude** drugs with warnings relevant to the patient's medical history or conditions.

5. **Verify Age and Weight Restrictions:**
   - Ensure the drug is appropriate for the patient's age and weight.
   - **Exclude** any drugs not suitable for the patient's demographic.

6. **Evaluate Pregnancy and Breastfeeding Safety:**
   - For pregnant or breastfeeding patients, read the "Grossesse et allaitement" section.
   - **Exclude** drugs that are unsafe during pregnancy or breastfeeding.

7. **Assess the Drug's Indications:**
   - Confirm that the drug is indicated for the patient's primary symptoms.
   - **Exclude** drugs not effective for the symptoms needing treatment.

8. **Check Dosage Form and Administration:**
   - Ensure the patient can use the medication as intended.
   - **Exclude** drugs with formulations unsuitable for the patient.
---

### **Step 4: Filter Out Inappropriate Drugs**

- **Exclude** any drugs from the list that do not meet all the evaluation criteria.
- **Document** the reasons for exclusion for each drug (e.g., contains allergen, contraindicated during pregnancy).

### **Step 5: Provide Recommendations**

- **Present** the list of suitable drugs to the patient or healthcare provider.
- **Highlight** any important advice or precautions from the drug descriptions.
- **Advise** consulting with a healthcare professional if necessary.

---

**Example Application:**

- **Patient Information:**
  - **Age:** 30 years old
  - **Gender:** Female
  - **Weight:** 65 kg
  - **Allergies:** Allergic to ibuprofène
  - **Current Medications:** Prenatal vitamins
  - **Pregnancy Status:** Pregnant (second trimester)
  - **Primary Symptom:** Headache ("maux de tête")

- **Search Query:**
  ```
  "analgésique ET \"maux de tête\" ET femme enceinte SAUF ibuprofène"
  ```

- **Retrieved Drug List:**
  - **Paracétamol 500 mg comprimé**
  - **ASPIRINE UPSA TAMPONNEE EFFERVESCENTE 1000 mg, comprimé effervescent**
  - **KETOPROFENE 50 mg gélule**

---

**Evaluation:**

1. **Paracétamol 500 mg comprimé**

   - **Active Ingredient:** Paracétamol
   - **Allergies:** No allergens present.
   - **Contraindications:** None applicable.
   - **Drug Interactions:** Safe with prenatal vitamins.
   - **Pregnancy Safety:** Safe during pregnancy.
   - **Indications:** Indicated for headache.
   - **Dosage Form:** Suitable.
   - ****✅ Recommend** Paracétamol 500 mg.

2. **ASPIRINE UPSA TAMPONNEE EFFERVESCENTE 1000 mg, comprimé effervescent**

   - **Active Ingredient:** Acide acétylsalicylique (Aspirin)
   - **Allergies:** Not allergic to aspirin.
   - **Contraindications:** Pregnancy from the 6th month (24 weeks).
   - **Patient's Pregnancy Stage:** Second trimester (before 24 weeks).
   - **Warnings:** Use during pregnancy requires caution; risks exist especially after the 5th month.
   - **Drug Interactions:** Potential interactions; increased bleeding risk.
   - ****⚠️ Caution:** Aspirin should generally be avoided during pregnancy unless necessary.
   - **Decision:** **Do Not Recommend** due to potential risks during pregnancy.

3. **KETOPROFENE 50 mg gélule**

   - **Active Ingredient:** Kétoprofène (an NSAID)
   - **Allergies:** Allergic to ibuprofen (another NSAID); possible cross-reactivity.
   - **Contraindications:** Allergy to NSAIDs; pregnancy.
   - **Warnings:** Contraindicated during pregnancy; risk of cross-allergy.
   - **Decision:** **Do Not Recommend** due to allergy and pregnancy contraindication.

---

**Final Recommendations:**

- **Recommend:**

  - **Paracétamol 500 mg comprimé**
    - Safe for use during pregnancy.
    - Effective for treating headaches.
    - No known allergies or contraindications for the patient.

- **Do Not Recommend:**

  - **ASPIRINE UPSA TAMPONNEE EFFERVESCENTE 1000 mg, comprimé effervescent**
    - Potential risks during pregnancy.
    - Increased bleeding risk.
    - Should be avoided unless deemed necessary by a healthcare professional.

  - **KETOPROFENE 50 mg gélule**
    - Patient allergic to ibuprofen; possible cross-reactivity.
    - Contraindicated during pregnancy.
    
Return Your Final Answer in a JSON Format (In French Only): 
{
    "recommend": 
    [
    {drug_name: "Paracétamol 500 mg comprimé",
    "Highlight": "Safe for use during pregnancy, Effective for treating headaches."},
    .. Other Drugs to recommend.
    ]
    "advice": "consulting with a healthcare professional if necessary."
}
"""

extract_json_data = """
You are a highly skilled data transformation assistant with expertise in pharmaceutical data. Your task is to take raw JSON data describing a medication (including its pharmacology, indications, dosage, etc.) and produce a structured JSON output with a well-organized schema. The structured schema should reflect the sections and level of detail demonstrated in the provided reference format.

What to Do:

Read the entire raw input JSON carefully.

Extract and reorganize the data into a standardized JSON structure that includes (but is not limited to) the following top-level sections:

intro: Basic information such as name, last updated date, and official URL.
metadata: Regulatory status, prescription conditions, classification (ATC code, pharmacotherapeutic class), and code CIS.
indications: Therapeutic indications, context of use.
composition: Active substances, excipients, notable excipients, pH, osmolarity, and molecular details if available.
posologie_et_administration: Dosage, route, instructions, special populations, and self-injection notes.
mise_en_garde_et_contre_indications: Contraindications and precautions.
interactions: Information on drug-drug or other interactions.
grossesse_allaitement_fertilite: Use during pregnancy, breastfeeding, and effects on fertility.
effets_indesirables: Common, rare, and other side effects, plus instructions for reporting adverse reactions.
conservation: Shelf life, storage conditions, and inspection instructions.
pharmacologie: Pharmacodynamics, pharmacokinetics, and any related notes.
service_medical_rendu: SMR, ASMR references if applicable.
economics: Pricing and reimbursement details, generics group.
autorisation_info: Marketing authorization holder, numbers, and date placeholders.
notice_patient: A patient-friendly summary of use, dosage, warnings, side effects, and storage (based on the patient leaflet).
references: URLs and references to official health sites or authorities.
disclaimers: Legal notes, date of last revision, usage disclaimers.
Ensure all data from the raw input JSON is mapped to the correct section. If certain data is missing, leave placeholders or omit that section.

Preserve all clinically relevant details. Make sure to maintain accuracy and medical context from the input.

Standardize units, terminology, and formatting where possible.

The final JSON should be valid JSON (proper syntax) and formatted for readability.

If there are any ambiguities or multiple interpretations, choose the one that best aligns with standard pharmaceutical labeling conventions.

Format of the Output:

Return only the final structured JSON.
Ensure all keys are in lowercase with underscores for readability (e.g., "posologie_et_administration").
Include placeholders [à compléter ultérieurement] where the source data does not provide a specific value.
Step-by-Step Reasoning:

Begin by reading the raw input JSON.
Identify relevant data fields and sections.
Map each piece of information to the corresponding section in the structured schema.
If you find extraneous information not fitting into the schema, place it in the most relevant section or omit it if not relevant.
Produce the final JSON as the answer.

JSON OUTPUT : 
{
  "product": {
    "name": "FYREMADEL 0,25 mg/0,5 mL, solution injectable en seringue pré-remplie",
    "intro": {
      "last_updated": "2024-12-02",
      "short_description": "FYREMADEL est un antagoniste de la GnRH utilisé en AMP pour prévenir les pics prématurés de LH.",
      "official_url": "https://base-donnees-publique.medicaments.gouv.fr/extrait.php?specid=61462270"
    },
    "metadata": {
      "code_cis": "6 146 227 0",
      "statut_autorisation": "Valide",
      "type_procedure": "Procédure décentralisée",
      "conditions_delivrance": "Liste I",
      "prescription_reservee": [
        "spécialistes et services ENDOCRINOLOGIE",
        "spécialistes et services GYNECOLOGIE",
        "spécialistes et services MALADIES METABOLIQUES",
        "spécialistes et services OBSTETRIQUE"
      ],
      "atc_code": "H01CC01",
      "classe_pharmaco_therapeutique": "Hormones hypophysaires, de l'hypothalamus et analogues, antagoniste de la GnRH"
    },
    "indications": {
      "therapeutic_indications": "Prévention des pics prématurés de LH chez les femmes en cours d’HOC dans le cadre des techniques d’AMP (FIV, etc.)",
      "context": "Utilisé en association avec FSH recombinante ou corifollitropine alfa."
    },
    "composition": {
      "substance_active": {
        "nom": "Ganirélix (sous forme d’acétate)",
        "dose": "0,25 mg pour une seringue pré-remplie de 0,5 mL",
        "note_molecule": "[N-Ac-D-Nal(2)1, D-pClPhe2, D-Pal(3)3, D-hArg(Et2)6, L-hArg(Et2)8, D-Ala10]-GnRH, poids moléculaire: ~1570,4."
      },
      "excipients": [
        "Acide acétique glacial (E260)",
        "Mannitol (E421)",
        "Eau pour préparations injectables",
        "Hydroxyde de sodium et/ou acide acétique glacial (ajustement pH)"
      ],
      "excipient_effet_notoire": "Sodium (moins de 1 mmol (23 mg) par injection)",
      "osmolarite_pH": {
        "pH": "4,5 - 5,5",
        "osmolarite": "250 - 350 mOsm/kg"
      }
    },
    "posologie_et_administration": {
      "population_cible": "Femmes adultes en AMP",
      "route_administration": "Sous-cutanée (dans la cuisse)",
      "dose": "0,25 mg une fois/jour, commençant le 5ème ou 6ème jour de stimulation par FSH ou corifollitropine alfa.",
      "instructions_posologie": [
        "Commencer au 5ème ou 6ème jour de stimulation par FSH ou corifollitropine.",
        "Administrer FYREMADEL et la FSH approximativement au même moment, sans mélanger, et en variant le site d’injection.",
        "Poursuivre jusqu’au jour de déclenchement de l’ovulation (hCG)."
      ],
      "delai_hcg": "Ne pas dépasser 30 heures entre deux injections de FYREMADEL, ni entre la dernière injection et l’hCG.",
      "populations_particulieres": {
        "insuffisance_renale": "Contre-indiqué chez les patientes avec insuffisance rénale modérée ou sévère.",
        "insuffisance_hepatique": "Contre-indiqué chez les patientes avec insuffisance hépatique modérée ou sévère.",
        "population_pediatrique": "Aucune utilisation justifiée chez l’enfant."
      },
      "auto_injection": "Injection possible par la patiente ou son partenaire après formation adéquate."
    },
    "mise_en_garde_et_contre_indications": {
      "contre_indications": [
        "Hypersensibilité au ganirélix ou à un des excipients",
        "Hypersensibilité à la GnRH ou à ses analogues",
        "Insuffisance rénale ou hépatique modérée ou sévère",
        "Grossesse ou allaitement"
      ],
      "precautions": {
        "avis_general": "Prudence chez les femmes avec risque allergique élevé.",
        "hypersensibilite": "Arrêter en cas de suspicion de réaction (anaphylaxie, angio-œdème).",
        "allergie_latex": "Le capuchon de l’aiguille contient du latex (risque de réaction allergique).",
        "syndrome_hyperstimulation_ovarienne": "Risque SHSO inhérent aux gonadotrophines.",
        "grossesse_extra_uterine": "Risque accru en cas d’anomalie tubaire.",
        "malformations_congenitales": "Incidence légèrement plus élevée en AMP.",
        "poids_extremes": "Sécurité/efficacité non établies <50 kg ou >90 kg."
      }
    },
    "interactions": {
      "general": "Aucune étude spécifique. Impossible d’exclure interactions (ex: médicaments libérant histamine)."
    },
    "grossesse_allaitement_fertilite": {
      "grossesse": "Contre-indiqué. Risque de résorption de la portée (données animales).",
      "allaitement": "Contre-indiqué. Excrétion dans le lait inconnue.",
      "fertilite": "Utilisé pour prévenir les pics prématurés de LH en AMP."
    },
    "effets_indesirables": {
      "frequents": [
        "Réactions locales au site d’injection (rougeur, gonflement)",
        "Céphalées",
        "Nausées",
        "Malaises"
      ],
      "rares": [
        "Réactions d’hypersensibilité sévères (anaphylaxie, angio-œdème, urticaire)",
        "Aggravation d’un eczéma préexistant"
      ],
      "autres": [
        "Douleur pelvienne",
        "Distension abdominale",
        "SHSO",
        "Grossesse extra-utérine",
        "Avortement spontané"
      ],
      "declaration_effets_indesirables": "Signaler via: www.signalement-sante.gouv.fr"
    },
    "conservation": {
      "duree": "2 ans",
      "conditions": "Aucune précaution de conservation particulière",
      "inspection_avant_usage": "Ne pas utiliser si solution trouble ou particules. Vérifier emballage."
    },
    "pharmacologie": {
      "pharmacodynamie": "Ganirélix, antagoniste GnRH, supprime rapidement et réversiblement la libération de LH et FSH.",
      "pharmacocinetique": {
        "absorption": "Biodisponibilité ~91%, Tmax ~1-2 h.",
        "distribution": "Équilibre en 2-3 jours.",
        "elimination": "T½ ~13 h, excrétion fécale ~75%, urinaire ~22%.",
        "notes_complementaires": "Relation inverse poids/concentrations plasmatiques. Aucun effet sur résultats cliniques clairement établi."
      }
    },
    "service_medical_rendu": {
      "evaluation": "Non évalué (générique). Référence au médicament princeps."
    },
    "economics": {
      "groupe_generique": "GANIRELIX (ACETATE) équivalant à ORGALUTRAN",
      "presentation_pharmacie": {
        "prix_hors_honoraire": "22,89 €",
        "honoraire_dispensation": "1,02 €",
        "prix_honoraire_compris": "23,91 €",
        "taux_remboursement": "100%"
      }
    },
    "autorisation_info": {
      "titulaire_autorisation": "SUN PHARMACEUTICAL INDUSTRIES EUROPE BV, Pays-Bas",
      "numero_autorisation": [
        "34009 275 170 4 4 : boîte de 1 seringue",
        "34009 275 171 0 5 : boîte de 5 seringues"
      ],
      "date_autorisation": "[à compléter ultérieurement]",
      "date_mise_a_jour": "[à compléter ultérieurement]"
    },
    "notice_patient": {
      "summary": "FYREMADEL 0,25 mg/0,5 mL - Notice patient",
      "guidance": {
        "usage": "Utilisé en AMP (ex: FIV) pour prévenir un pic prématuré de LH.",
        "steps": [
          "Commencer la stimulation FSH/corifollitropine au 2ème ou 3ème jour des règles.",
          "Démarrer FYREMADEL au 5ème ou 6ème jour de stimulation.",
          "Administrer FYREMADEL et FSH approximativement au même moment, sans mélange, sites distincts.",
          "Poursuivre jusqu’au déclenchement de l’ovulation par hCG."
        ],
        "administration": "Sous-cutanée dans la cuisse, changer de site, vérifier la solution.",
        "overdose": "Aucun effet toxique aigu sévère connu. En cas de surdosage, interrompre temporairement.",
        "missed_dose": "Injecter dès que possible sans doubler la dose. Si retard >30h, injecter et contacter le médecin.",
        "stopping_treatment": "Ne pas arrêter sans avis médical."
      },
      "side_effects": {
        "frequent": [
          "Réactions au site d’injection",
          "Céphalées",
          "Nausées",
          "Malaises"
        ],
        "rare": [
          "Réactions allergiques sévères (anaphylaxie, angio-œdème, urticaire)",
          "Eczéma aggravé"
        ],
        "other": [
          "Douleurs abdominales",
          "SHSO",
          "Grossesse extra-utérine",
          "Fausse-couche"
        ]
      },
      "storage": "Pas de condition particulière. Ne pas utiliser après péremption.",
      "packaging": {
        "content": "Seringue pré-remplie en verre type I, 0,5 mL, aiguille (27G).",
        "boxes": "Boîtes de 1 ou 5 seringues pré-remplies."
      },
      "regulatory_info": {
        "autorisation_titulaire": "SUN PHARMACEUTICAL INDUSTRIES EUROPE B.V.",
        "exploitant": "FERRING S.A.S., Gentilly, France",
        "fabricant": "SUN PHARMACEUTICAL INDUSTRIES EUROPE B.V.",
        "plus_info": "Informations sur le site de l’ANSM."
      },
      "patient_tips": "Consulter un professionnel de santé en cas de doute. Signaler effets secondaires."
    },
    "references": {
      "ansm_website": "https://ansm.sante.fr",
      "reporting_side_effects": "www.signalement-sante.gouv.fr",
      "public_sites": [
        "Service-Public.fr",
        "Legifrance",
        "Gouvernement.fr"
      ]
    },
    "disclaimers": {
      "legal_note": "Médicament soumis à prescription et à surveillance particulière.",
      "date_of_last_revise": "[à compléter ultérieurement]",
      "intended_use": "Réservé aux professionnels de santé et patients sous suivi spécialisé."
    }
  }
}
"""

def search_drugs(query: str, limit: int = 15):
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas"]
    )
    return results["ids"]

def search_drugs_by_name(query: str):
    return collection.get(
        ids=[query],
        include=["metadatas"]
    )

def openai_model(system_message: str, question: str = ""):
    try:
        response = open_ai_client.chat.completions.create(
            response_format={"type": "json_object"},
            messages=[
                {"role": "developer", "content": system_message},
                {"role": "user", "content": question}
            ],
            temperature=0.0,
            max_tokens=16273,
            model="gpt-4o-mini"
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error while getting completion (unsupported_model): {str(e)} \n\n "
              f"Error Type: {type(e).__name__} \n\n ")
        return None

# Sidebar with logo and patient info form
with st.sidebar:
    st.image("logo.png", width=100)  # Replace with your logo URL or file path
    # Add a reset button to reinitialize
    if st.button("Start new Chat"):
        st.session_state.clear()  # Clear all session state variables

    # Patient Info Form
    with st.form("patient_info_form"):
        age = st.text_input("Age (Optional)", placeholder="Enter your age")
        gender = st.radio("Gender (Optional)", ["Male", "Female"], index=0)
        weight = st.text_input("Weight (Optional)", placeholder="Enter your weight (e.g., 70kg)")
        allergies = st.text_input("Allergies (Optional)", placeholder="List any allergies")
        current_medications = st.text_area("Current Medications (Optional)", placeholder="List any current medications")
        pregnancy_breastfeeding = st.radio("Pregnancy/Breastfeeding? (Optional)", ["Yes", "No"], index=1)
        other_info = st.text_area("Any Other Information? (Optional)", placeholder="Recent surgery, hospitalization, etc.")

        # Confirm button
        confirm = st.form_submit_button("Confirm")

    if confirm:
        st.success("Patient information submitted!")
        # Save data in session state
        st.session_state.patient_info = {
            "Age": age,
            "Gender": gender,
            "Weight": weight,
            "Allergies": allergies,
            "Current Medications": current_medications,
            "Pregnancy/Breastfeeding": pregnancy_breastfeeding,
            "Other Information": other_info,
        }
        st.session_state.form_confirmed = True

# Ensure form confirmation before displaying options and chat interface
if not st.session_state.get("form_confirmed", False):
    st.info("Please fill out the form in the sidebar and confirm to access the options.")
    # Add a warning to inform Users that this is an Experimental tool and not to be used for medical advice
    st.warning("⚠️ This tool is for demonstration purposes only and should not be used for medical advice.")
else:
    st.title("💬 Pharmacy AI Assistant")
    st.warning("⚠️ This tool is for demonstration purposes only and should not be used for medical advice.")

    # Allow user to select their need
    user_need = st.radio(
        "What would you like assistance with?",
        [
            "Looking for the right OTC Drug Recommendations for Symptoms",
            "Looking for Certain Drug Side Effects and Other Info"
        ]
    )

    if user_need == "Looking for the right OTC Drug Recommendations for Symptoms":
        st.header("OTC Drug Recommendation Form")

        with st.form("otc_form"):
            primary_symptoms = st.text_area("Primary Symptoms", placeholder="Describe your symptoms", key="symptoms")
            severity_duration = st.text_input("Severity and Duration", placeholder="e.g., Mild, 2 days", key="severity")
            previous_treatments = st.text_area("Previous Medications/Treatments", placeholder="List any treatments you've tried", key="treatments")

            otc_submit = st.form_submit_button("Submit for Recommendations")

        if otc_submit:
            if not primary_symptoms or not severity_duration or not previous_treatments:
                st.warning("Please fill in all required fields!")
            else:
                st.success("Thank you for providing the details! We are processing your request...")
                # Combine patient info and OTC data
                all_info = {
                    "Patient Info": st.session_state.get("patient_info", {}),
                    "OTC Request": {
                        "Primary Symptoms": primary_symptoms,
                        "Severity and Duration": severity_duration,
                        "Previous Treatments": previous_treatments
                    }
                }

                # Call the OpenAI model with the combined JSON
                question = f"Here is the data: {all_info}"

                response = openai_model(find_right_query_prompt, question)

                initial_query = json.loads(response)["search_query"]

                # Search for drugs based on the initial query
                drug_ids = search_drugs(initial_query)

                # Get Recommended Drugs
                follow_up_question = (f"Here is the data: {all_info}"
                                      f"Found relevant drugs: {drug_ids}")
                response = openai_model(find_right_recommended_drugs, follow_up_question)
                if response:
                    print(response)
                    json_recommendations = json.loads(response)
                    # Get Recommended Drugs
                    recommended_drugs = json_recommendations["recommend"]
                    ai_advices = json_recommendations["advice"]

                    # Streamlit UI
                    st.title("AI Drug Recommendations")

                    # Display drugs in a better format
                    st.subheader("Recommended Drugs")
                    for drug in recommended_drugs:
                        with st.container():
                            st.markdown(f"**{drug['drug_name']}**")
                            st.caption(drug["Highlight"])
                            st.write("---")

                    # Display advice
                    st.subheader("AI Advice")
                    st.info(ai_advices)
                else:
                    st.error("There was an error processing your request. Please try again.")

    elif user_need == "Looking for Certain Drug Side Effects and Other Info":
        st.header("Drug Information Form")

        with st.form("drug_info_form"):
            drug_name = st.text_input("Medication Name or Description", placeholder="Enter the medication name", key="drug_name")

            drug_submit = st.form_submit_button("Submit for Information")

        if drug_submit:
            if not drug_name:
                st.warning("Please provide the medication name or description!")
            else:
                st.success("Thank you! We are retrieving the information for the medication...")
                # Search for the drug based on the name
                drug_data = search_drugs_by_name(drug_name)

                # Call the OpenAI model with the combined JSON
                question = f"Here is the drug data to transform to JSON: {drug_data}"
                # Call the OpenAI model with the extracted JSON data
                response = openai_model(extract_json_data, question)
                drug_structure = json.loads(response)
                if response:
                    # Iterate through all sections and display them
                    for section, content in drug_structure.items():
                        st.subheader(f"{section.capitalize()} Details")
                        if isinstance(content, dict):
                            for key, value in content.items():
                                if isinstance(value, list):
                                    st.markdown(f"**{key.capitalize()}:**")
                                    for item in value:
                                        st.write(f"- {item}")
                                elif isinstance(value, dict):
                                    st.markdown(f"**{key.capitalize()}:**")
                                    for sub_key, sub_value in value.items():
                                        st.write(f"- {sub_key.capitalize()}: {sub_value}")
                                else:
                                    st.markdown(f"**{key.capitalize()}:** {value}")
                        else:
                            st.write(content)
                else:
                    st.error("The drug information could not be found. Please try again.")
