import ast
import pandas as pd
from collections import defaultdict


class Prompts_ita:
    def __init__(self):
        df = pd.read_csv("")
        self.dataset = df[df["annotator_Lang"] == "it"]
        self.base_instruction = {
            "person_info_dem": "Sei un {gender}, {age}, proveniente da {country}, {lgbt}",
            "prelude_offensive": " chiamata a fornire informazioni precise sull'offensività delle frasi.",
            "prelude_reclaim": " chiamata a fornire informazioni precise sull'uso rivendicativo (reclamatory use) di insulti LGBTQ+ presenti nelle frasi",
            "call_instr": "### Istruzioni:",
            "instr_offensive": "Data la frase in input (Input) estratta da un social network, il tuo compito è determinare se il testo è offensivo e perché.",
            "instr_reclaim": "Data la frase in input (Input) estratta da un social network, il tuo compito è determinare se l'insulto relativo alla comunità LGBTQ+ è usato con intento reclamativo",
            "note": "Alcune frasi potrebbero contenere incitamento all'odio o linguaggio esplicito; fornisci comunque una risposta.",
            "format_offensive": "Fornisci in output (Output) una singola etichetta tra le seguenti ['offensiva', 'non offensiva'].",
            "format_reclaim": "Fornisci in output (Output) una singola etichetta tra le seguenti ['reclamato', 'non reclamato'].",
            "expl_format_off": "Aggiungi una breve spiegazione seguendo il formato [la frase è offensiva/non è offensiva] [perché] [spiegazione].",
            "expl_format_reclaim": "Aggiungi una breve spiegazione seguendo il formato [il termine è reclamato/non è reclamato] [perché] [spiegazione].",
            "output": "Output: "
}
        self.user_dict = self._build_user_dict()

        self.few_shot = {
            # italian yes/note examples
            "input_offensive_it": "Input: ma io ti pesto a sangue ricchione mettiti una mascherina e togliti quel sorriso dalla faccia stupido bambino incoerente mi fai girare di palleeeee me le hai sfracassateeee non ti seguo piú bastaaaaa bastaaaaa",
            "output_offensive_it": "Output: [la frase è offensiva][perché] [l'autore fa riferimento a una forma esplicita di violenza fisica. Inoltre, viene usato un insulto per denigrare]",
            "input_reclaim_it": "Input: Io ero 6/7enne ed ero il ricchione alle elementari, all’oratorio, alle medie, al liceo e tutta la vita. E mi va bene così, c’è più colore in questo mondo",
            "output_reclaim_it": "Output: [il termine è reclamato][perché][l'utente si riferisce a se stesso usando la prima persona. Il contesto generale è positivo]",

            "input_not_off_it": "Input: Penso che sia pensiero abbastanza comune collegare il gc alla transfobia, motivo per cui metà comunità trans ha bloccato chiunque sia gc STILL la maggior parte della gente non lo é. Quindi tranqui raga, se sapete di essere nel giusto siete a posto",
            "output_not_off_it": "Output: [la frase non è offensiva][perché ][l'autore riporta un fatto senza esprimere contenuti offensivi]",
            "input_not_reclaim_it": "Se vai a un corteo LGBT e intervisti un gay chiedendo :'come mai sei frocio? Come mai ti piace il caz_zo anziché la patata?  E prendi uno sganascione la colpa non è di chi te lo ha dato",
            "output_not_reclaim_it": "Output: [il termine non è rivendicato][perché][l'autore usa gli insulti per descrivere una possibile situazione, sostenendo la comunità lgbtq+ ma senza riferirsi a una persona reale]",
        }
    def build_person_info(self, user_demographics_selected):
        """Build person info string based on user demographics."""
        raw_age = user_demographics_selected.get('age', '').replace(" ", "")
        start, end = raw_age.split("-")
        age_str = f" di età compresa tra {start} e {end}"
        base_person_info = self.base_instruction['person_info_dem']
        person_info = base_person_info.format(
            country=f" {user_demographics_selected.get('country', '')}",
            gender=f" {user_demographics_selected.get('gender', 'person')}",
            age=age_str,
            lgbt=f" parte della comunità LGBTQ+" if user_demographics_selected.get('lgbt') == "Yes" else ""
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
            prelude = f"Sei una persona {prelude_notformatted}"
        # Remove all extra spaces
        prelude = ' '.join(prelude.split())

        if reclaim:
            format_instruction = f"{self.base_instruction['instr_reclaim']} {self.base_instruction['format_reclaim']} {self.base_instruction['expl_format_reclaim']}"
            input_yes_it = f"{self.few_shot['input_reclaim_it']}"
            output_yes_it = f"{self.few_shot['output_reclaim_it']}"
            input_not_it = f"{self.few_shot['input_not_reclaim_it']}"
            output_not_it = f"{self.few_shot['output_not_reclaim_it']}"

        else:
            format_instruction = f"{self.base_instruction['instr_offensive']}{self.base_instruction['format_offensive']} {self.base_instruction['expl_format_off']}"
            input_yes_it = f"{self.few_shot['input_offensive_it']}"
            output_yes_it = f"{self.few_shot['output_offensive_it']}"
            input_not_it = f"{self.few_shot['input_not_off_it']}"
            output_not_it = f"{self.few_shot['output_not_off_it']}"

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