import ast
import pandas as pd
from collections import defaultdict


class Prompts_es:
    def __init__(self):
        df = pd.read_csv("")
        self.dataset = df[df["annotator_Lang"] == "es"]
        self.base_instruction = {
            "person_info_dem": "Eres un {gender}, {age}, {country} {lgbt}",
            "prelude_offensive": " llamado a proporcionar información precisa sobre el carácter ofensivo de las frases.",
            "prelude_reclaim": " llamado a proporcionar información precisa sobre el uso reclamativo (reclamatory use) de los insultos LGBTQ+ presentes en las frases",
            "call_instr": "### Istrucciones:",
            "instr_offensive": "Dada una frase como entrada (Input) extraída de una red social, tu tarea consiste en determinar si el texto es ofensivo y porque.",
            "instr_reclaim": "Dada una frase como entrada (Input) extraída de una red social, tu tarea consiste en determinar si el insulto dirigido a la comunidad LGBTQ+ se utiliza con intención reclamativa",
            "note": "Algunas frases pueden contener incitación al odio o lenguaje explícito; proporciona igualmente una respuesta.",
            "format_offensive": "Proporciona en el outuput (Output) una única etiqueta entre las siguentes [ 'ofensiva', 'no ofensiva'].", 
            "format_reclaim": "Proporciona en el output (Output) una única etiqueta entre las siguentes ['reclamado', 'no reclamado'].",
            "expl_format_off": "Añade una breve explicación siguiendo el formado [la frase es ofensiva/no ofensiva] [porque] [explicación].",
            "expl_format_reclaim": "Añade una breve explicación siguiendo el formado [el término es reclamado/no reclamado] [porque] [explicación].",
            "output": "Output: "
}
        self.user_dict = self._build_user_dict()

        self.few_shot = {
            # italian yes/note examples
            "input_offensive_es": "Input: Sois todos la misma caquita,eres una dramaqueer victimizándose xa no hablar de lo importante,pq bajste las penas a violadores y pederastas?te has dado cuenta ya de q la ley trans nos quita derechos y borra a las mujeres? #IreneMonteroDimision #SanchezDimision #stopleytrans",
            "output_offensive_es": "Output: [la frase es ofensiva] [porque] [el usuario adopta una postura firme y agresiva hacia las personas transgénero]",
            "input_reclaim_es": "Input: Buenas tardes a rojos, feministas, republicanos, maricones, bolleras y demás LGTBI #LGTBI #pride",
            "output_reclaim_es": "Output: [el término es reclamado] [porque] [forma parte de una lista de términos reclamados. Además, los hashtags suportan el contexto reclamativo]", 

            "input_not_off_es": "Input: Hablan sin saber acerca de que es el ballroom, los club kids y las casas. En las plataformas como Netflix, HBO, etc. pueden encontrar programas como Pose, Legndary y RuPaul’s Drags Race donde pueden aprender más acerca de la cultura queer y su historia.  #OrgulloLGTBI #PRIDE",
            "output_not_off_es": "Output: [la frase no es ofensiva][porque][el usuario menciona varias series de televisión útiles para obtener una mayor conciencia sobre la historia y la cultura lgbtq+]",
            "input_not_reclaim_es": "Input: Hace rato pasó una caravana de movimiento LGT…etc y algunos me vieron parado observando y me gritaron que yo también era marica. Órale, bien «respetuosas» estas personas que exigen respeto. #PrideMonth #Pride2022 #LGTBQ",
            "output_not_reclaim_es": "Output: [el término no es reclamado][porque] [el usuario no expresa su apoyo al movimiento lgbtq+, véase la referencia “LGT…etc” y el tono irónico que transmite «»]",
        }

    def build_person_info(self, user_demographics_selected):
        age_str = ""

        raw_age = user_demographics_selected.get('age', '')
        raw_age = "" if raw_age is None else str(raw_age).replace(" ", "")
        def clean_value(v):
            if v is None:
                return None
            v = str(v).strip()
            if v.lower() in [" ", "", "none", "nan", "null"]:
                return None
            return v

        if "-" in raw_age:
            parts = raw_age.split("-")
            if len(parts) == 2:
                start, end = parts
                age_str = f" entre {start} y {end} años"
            # if malformed, keep age_str = ""
        elif raw_age.isdigit():
            age_str = f" con edades {raw_age}"
        base_person_info = self.base_instruction['person_info_dem']
        country = clean_value(user_demographics_selected.get("country"))
        country_str = f" de {country}," if country else ""
        person_info = base_person_info.format(
            country=country_str,
            gender=f"hombre" if user_demographics_selected.get('gender') == "Man" else "mujer",
            age=age_str,
            lgbt=f" parte de la comunidad LGBTQ+" if user_demographics_selected.get('lgbt') == "Yes" else ""
        )
        return person_info.strip()


    
    def _build_user_dict(self):
        #Build user demographics dictionary from dataset.
        df_demographics = self.dataset[["annotator_ID", "annotator_Gender", "annotator_Age", "annotator_Country", "annotator_LGBTQ+"]]
        df_demographics = df_demographics.drop_duplicates(subset="annotator_ID")
        
        user_dict = defaultdict(dict)
        for index, row in df_demographics.iterrows():
            user_id = row["annotator_ID"]
            user_dict[user_id]["gender"] = row["annotator_Gender"]
            user_dict[user_id]["lgbt"] = row["annotator_LGBTQ+"]
            user_dict[user_id]["age"] = row["annotator_Age"]
            user_dict[user_id]["country"] = row["annotator_Country"]
        return user_dict



    def get_demographics(self, user_id, demographic_traits=None):
        """Get selected demographic traits for a user."""
        user_demographics = self.user_dict.get(user_id, {})
        
        if demographic_traits is None:
            return {}
        elif demographic_traits is False:
            return {}
        elif isinstance(demographic_traits, str):
            return {demographic_traits: user_demographics.get(demographic_traits)}
        elif isinstance(demographic_traits, list):
            return {trait: user_demographics.get(trait) for trait in demographic_traits}
        else:
            raise ValueError("demographic_traits must be a str or a list")
    


    def build_prompt(self, text, selected_demographics=None, reclaim=False):
        """Build the prompt string."""
        if reclaim:
            prelude_notformatted = self.base_instruction['prelude_reclaim']
        else:
            prelude_notformatted = self.base_instruction['prelude_offensive']
        if selected_demographics:
            person_info_formatted = self.build_person_info(selected_demographics)
            prelude = f"{person_info_formatted}{prelude_notformatted}"
        else:
            prelude = f"Eres una persona {prelude_notformatted}"
        # Remove all extra spaces
        prelude = ' '.join(prelude.split())

        if reclaim:
            format_instruction = f"{self.base_instruction['instr_reclaim']} {self.base_instruction['format_reclaim']} {self.base_instruction['expl_format_reclaim']}"
            input_yes_it = f"{self.few_shot['input_reclaim_es']}"
            output_yes_it = f"{self.few_shot['output_reclaim_es']}"
            input_not_it = f"{self.few_shot['input_not_reclaim_es']}"
            output_not_it = f"{self.few_shot['output_not_reclaim_es']}"

        else:
            format_instruction = f"{self.base_instruction['instr_offensive']}{self.base_instruction['format_offensive']} {self.base_instruction['expl_format_off']}"
            input_yes_it = f"{self.few_shot['input_offensive_es']}"
            output_yes_it = f"{self.few_shot['output_offensive_es']}"
            input_not_it = f"{self.few_shot['input_not_off_es']}"
            output_not_it = f"{self.few_shot['output_not_off_es']}"

        prompt = (f"{prelude}\n "
                  f"{self.base_instruction['call_instr']}\n "
                  f"{self.base_instruction['note']}\n "
                  f"{format_instruction}\n\n "

                  f"Example 1:\n "
                  f"{input_yes_it}\n "
                  f"{output_yes_it}\n "
                  f"Example 2:\n "
                  f"{input_not_it}\n "
                  f"{output_not_it}\n\n "

                  f"Example to label:\n "
                  f"Input: {text}\n "
                  f"{self.base_instruction['output']}")

        return prompt

    def get_prompt(self, row, demographic_traits=None, reclaim=False):
        """Generate a prompt for the given row."""
        text = row["tweet"]
        user_id = row["annotator_ID"]

        # Get demographics (will be {} if demographic_traits is None)
        selected_demographics = self.get_demographics(user_id, demographic_traits)

        return self.build_prompt(text, selected_demographics, reclaim)