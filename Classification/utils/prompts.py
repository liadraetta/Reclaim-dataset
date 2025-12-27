import ast
import pandas as pd
from collections import defaultdict


class Prompts:
    def __init__(self):
        self.dataset = pd.read_csv("")
        self.base_instruction = {
            "person_info_dem": "You are a {gender}, {age},{country} {lgbt}",
            "prelude_offensive": " asked to provide precise information about offensiveness of sentences.",
            "prelude_reclaim": " asked to provide precise information about the reclamatory use of LGBTQ+ slur present in the sentences",
            "call_instr": "### Instruction:",
            "instr_offensive": "Given the sentence in input (Input) taken from a social network, your task is to determine if the text is offensive and why.",
            "instr_reclaim": "Given the sentence in input (Input) taken from a social network, your task is to determine if the slur related to the LGBTQ+ community is used with a reclamatory intent",
            "note": "Some of the sentences may contain hate or explicit language, please provide an answer in any case.",
            "format_offensive": "Provide in output (Output) a single label among the following arrays ['offensive', 'not offensive'].",
            "format_reclaim": "Provide in output (Output) a single label among the following arrays ['reclaimed', 'not reclaimed'].",
            "expl_format_off": "Add a brief explanation following the format [the sentence is offensive/is not offensive] [because] [explanation].",
            "expl_format_reclaim": "Add a brief explanation following the format [the term is reclaimed/is not reclaimed] [because] [explanation].",
            "output": "Output: ",
        }
        self.user_dict = self._build_user_dict()

        self.few_shot = {
            # italian yes/note examples
            "input_offensive_it": "Input: ma io ti pesto a sangue ricchione mettiti una mascherina e togliti quel sorriso dalla faccia stupido bambino incoerente mi fai girare di palleeeee me le hai sfracassateeee non ti seguo piú bastaaaaa bastaaaaa",
            "output_offensive_it": "Output: [the sentence is offensive][because] [the writer refers to an explicitly form of physical violence. In addition, a slur is used to denigrate]",
            "input_reclaim_it": "Input: Io ero 6/7enne ed ero il ricchione alle elementari, all’oratorio, alle medie, al liceo e tutta la vita. E mi va bene così, c’è più colore in questo mondo 🌈",
            "output_reclaim_it": "Output: [the term is reclaimed][because][ the user refers to himself by using the first person. The overall context is positive",

            "input_not_off_it": "Input: Penso che sia pensiero abbastanza comune collegare il gc alla transfobia, motivo per cui metà comunità trans ha bloccato chiunque sia gc STILL la maggior parte della gente non lo é. Quindi tranqui raga, se sapete di essere nel giusto siete a posto",
            "output_not_off_it": "Output: [the sentence is not offensive][because the writer reports a fact without expressing offensive content]",
            "input_not_reclaim_it": "Se vai a un corteo LGBT e intervisti un gay chiedendo :'come mai sei frocio? Come mai ti piace il caz_zo anziché la patata?  E prendi uno sganascione la colpa non è di chi te lo ha dato",
            "output_not_reclaim_it": "Output: [the term is not reclaimed][because][the writer adopts the slurs to describe a possible situation, supporting the lgbtq+ community but without referring to real person]",
            # english yes/note examples
            "input_offensive_en": "Input: no you didnt faggot. what the fuck are you trying to pull queer? you little piece of shit reposter go kill your self i mean youre a total dumbass the post are timestamped you silly fuck",
            "output_offensive_en": "Output: [the sentence is offensive][because][the writer uses a lot of slurs and insults to express an offensive thought]",
            "input_reclaim_en": "Input: I've always made ownership of the word 'fag'—I use it as a badge of honor. When someone yelled 'faggot' at me the other day, I smiled and yelled back 'true!' because I didn't feel anything bad about the word, even if they intended it that way.",
            "output_reclaim_en": "Output: [the term is reclaimed][because][the user explicitly choose to adopt “fag” to express themselves, stripping the term of its offensive power]",

            "input_not_off_en": "Input: As a younger closeted guy I said faggot all the time. Not proud of it but yeah....",
            "output_not_off_en": "Output: [the sentence is not offensive][because][it only conveys a personal habits]",
            "input_not_reclaim_en": "Input: The issue with that line is that it makes it seem like being a 'faggot' is a bad thing. Your rebuttals, or at least your example, only justifies and makes their behavior and language ok.",
            "output_not_reclaim_en": "Output: [the term is not reclaimed][because][ the user only proposes personal consideration about being a faggot without taking strong position]",
            # spanish yes/note examples
            "input_offensive_es": "Input: Sois todos la misma caquita,eres una dramaqueer victimizándose xa no hablar de lo importante,pq bajste las penas a violadores y pederastas?te has dado cuenta ya de q la ley trans nos quita derechos y borra a las mujeres? #IreneMonteroDimision #SanchezDimision #stopleytrans",
            "output_offensive_es": "Output: [the sentence is offensive][because][ the user takes a strong and aggressive position against trans people]",
            "input_reclaim_es": "Input: Buenas tardes a rojos, feministas, republicanos, maricones, bolleras y demás LGTBI #LGTBI #pride",
            "output_reclaim_es": "Output: [the term is reclaimed][beacuse][it is part of a list of reclaimed terms. Moreover, the hashtags support the reclamatory context]",

            "input_not_off_es": "Input: Hablan sin saber acerca de que es el ballroom, los club kids y las casas. En las plataformas como Netflix, HBO, etc. pueden encontrar programas como Pose, Legndary y RuPaul’s Drags Race donde pueden aprender más acerca de la cultura queer y su historia.  #OrgulloLGTBI #PRIDE",
            "output_not_off_es": "Output: [the sentence is not offensive][because][ the user lists various tv series useful to become more aware about lgbtq+ history and culture]",
            "input_not_reclaim_es": "Input: Hace rato pasó una caravana de movimiento LGT…etc y algunos me vieron parado observando y me gritaron que yo también era marica. Órale, bien «respetuosas» estas personas que exigen respeto. #PrideMonth #Pride2022 #LGTBQ",
            "output_not_reclaim_es": "Output: [the term is not reclaimed][because] [the writer does not express support to the lgbtq+ movement, see the reference “LGT…etc” and the ironic tone conveyed by «»]",
        }
    def build_person_info(self, user_demographics_selected):
        """Build person info string based on user demographics."""
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
                age_str = f" aged between {start} and {end}"
            # if malformed, keep age_str = ""
        elif raw_age.isdigit():
            # Single age (e.g., "30")
            age_str = f" aged {raw_age}"
        base_person_info = self.base_instruction['person_info_dem']
        country = clean_value(user_demographics_selected.get("country"))
        country_str = f" from {country}," if country else ""
        person_info = base_person_info.format(
            country=country_str,
            gender=f" {user_demographics_selected.get('gender', 'person')}",
            age=age_str,
            lgbt=f" part of the LGBTQ+ community" if user_demographics_selected.get('lgbt') == "Yes" else ""
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
            prelude = f"You are a person {prelude_notformatted}"
        # Remove all extra spaces
        prelude = ' '.join(prelude.split())

        if reclaim:
            format_instruction = f"{self.base_instruction['instr_reclaim']} {self.base_instruction['format_reclaim']} {self.base_instruction['expl_format_reclaim']}"
            input_yes_it = f"{self.few_shot['input_reclaim_it']}"
            output_yes_it = f"{self.few_shot['output_reclaim_it']}"
            input_not_it = f"{self.few_shot['input_not_reclaim_it']}"
            output_not_it = f"{self.few_shot['output_not_reclaim_it']}"
            input_yes_en = f"{self.few_shot['input_reclaim_en']}"
            output_yes_en = f"{self.few_shot['output_reclaim_en']}"
            input_not_en = f"{self.few_shot['input_not_reclaim_en']}"
            output_not_en = f"{self.few_shot['output_not_reclaim_en']}"
            input_yes_es = f"{self.few_shot['input_reclaim_es']}"
            output_yes_es = f"{self.few_shot['output_reclaim_es']}"
            input_not_es = f"{self.few_shot['input_not_reclaim_es']}"
            output_not_es = f"{self.few_shot['output_not_reclaim_es']}"

        else:
            format_instruction = f"{self.base_instruction['instr_offensive']}{self.base_instruction['format_offensive']} {self.base_instruction['expl_format_off']}"
            input_yes_it = f"{self.few_shot['input_offensive_it']}"
            output_yes_it = f"{self.few_shot['output_offensive_it']}"
            input_not_it = f"{self.few_shot['input_not_off_it']}"
            output_not_it = f"{self.few_shot['output_not_off_it']}"
            input_yes_en = f"{self.few_shot['input_offensive_en']}"
            output_yes_en = f"{self.few_shot['output_offensive_en']}"
            input_not_en = f"{self.few_shot['input_not_off_en']}"
            output_not_en = f"{self.few_shot['output_not_off_en']}"
            input_yes_es = f"{self.few_shot['input_offensive_es']}"
            output_yes_es = f"{self.few_shot['output_offensive_es']}"
            input_not_es = f"{self.few_shot['input_not_off_es']}"
            output_not_es = f"{self.few_shot['output_not_off_es']}"

        prompt = (f"{prelude}\n "
                  f"{self.base_instruction['call_instr']}\n "
                  f"{self.base_instruction['note']}\n "
                  f"{format_instruction}\n\n "

                  f"Example 1:\n "
                  f"{input_yes_it}\n "
                  f"{output_yes_it}\n "
                  f"Example 2:\n "
                  f"{input_not_it}\n "
                  f"{output_not_it}\n "
                  f"Example 3:\n "
                  f"{input_yes_en}\n "
                  f"{output_yes_en}\n "
                  f"Example 4:\n "
                  f"{input_not_en}\n "
                  f"{output_not_en}\n "
                  f"Example 5:\n "
                  f"{input_yes_es}\n "
                  f"{output_yes_es}\n "
                  f"Example 6:\n "
                  f"{input_not_es}\n "
                  f"{output_not_es}\n\n "

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