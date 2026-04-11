import pandas as pd
import numpy as np
import ast
import re
from sklearn.model_selection import train_test_split


class NLPPreprocessing:
    def __init__(self, df, columns, column_removal_threshold):
        self.df = df[columns].copy()
        self.column_removal_threshold = column_removal_threshold
        self.bad_values = {
            'None', 'N/A', 'City', 'State', 'Province', '', 'nan', None,
            'N, A', 'City , State', 'City, State', 'Company Name',
            'Company Name ï¼ City , State', 'Company Name ï¼ City', 'Company Name ï¼'
        }
        self.delimiters = {"\n", ";", "|", "\t", "•", "·"}

        self.candidate_fields =[
            f for f in [
            "career_objective",
            "skills",
            "degree_names",
            "major_field_of_studies",
            "educational_institution_name",
            "positions",
            "professional_company_names",
            "responsibilities",
            ] if f in columns
        ]
        self.job_fields = [
            f for f in [
            "job_position_name",
            "skills_required",
            "responsibilities.1",
            "educationaL_requirements",
            "experiencere_requirement",
         ] if f in columns
        ]

    def preprocess_and_split(self, test_size=0.2):
        self.df = self.clean_and_remove_columns()
        self.df = self.impute()
        self.df = self.consolidate()
        cols_to_keep = [
            "candidate_consolidated",
            "job_consolidated",
            "job_position_name",
            "matched_score",
        ]
        df_final = self.df[[c for c in cols_to_keep if c in self.df.columns]]  # fix here
        X = df_final.drop(columns='matched_score')
        y = df_final['matched_score']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=X["job_position_name"]
        )
        # Drop job_position_name after stratification
        X_train = X_train.drop(columns="job_position_name")
        X_test = X_test.drop(columns="job_position_name")
        return X_train, X_test, y_train, y_test


    @staticmethod
    def _field_to_text(val, max_words=None):
        """
        Convert a cleaned field value (list, str, or scalar) into a plain string.
        Filters out 'unknown' placeholders left by impute(). Optionally truncates.
        """
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        if isinstance(val, list):
            text = ", ".join(
                str(v).strip() for v in val
                if str(v).strip().lower() != "unknown"
            )
        else:
            text = str(val).strip()
            if text.lower() == "unknown":
                return ""

        text = re.sub(r'\s+', ' ', text).strip()

        if max_words:
            words = text.split()
            text = " ".join(words[:max_words])

        return text

    def _build_consolidated(self, row, fields, max_words=200):
        """
        Join whichever fields exist in the row into a labelled text string.
        Format: "field_label: value | field_label: value | ..."
        Total output is truncated to max_words to respect SBERT's 256-token limit.
        """
        parts = []
        for field in fields:
            if field not in row.index:
                continue
            label = field.replace("responsibilities.1", "job_responsibilities") \
                         .replace("educationaL_requirements", "education") \
                         .replace("experiencere_requirement", "experience") \
                         .replace("skills_required", "skills") \
                         .replace("job_position_name", "title") \
                         .replace("career_objective", "objective") \
                         .replace("major_field_of_studies", "major") \
                         .replace("educational_institution_name", "institution") \
                         .replace("professional_company_names", "companies") \
                         .replace("degree_names", "degree")
            text = self._field_to_text(row[field])
            if text:
                parts.append(f"{label}: {text.lower()}")

        combined = " | ".join(parts)
        words = combined.split()
        return " ".join(words[:max_words])

    def consolidate(self):
        """
        Adds two new columns to the DataFrame:
          - candidate_consolidated : merged candidate profile text
          - job_consolidated       : merged job posting text
        Only uses fields that are present in the DataFrame after cleaning.
        """
        available_candidate = [f for f in self.candidate_fields if f in self.df.columns]
        available_job       = [f for f in self.job_fields       if f in self.df.columns]

        self.df["candidate_consolidated"] = self.df.apply(
            lambda row: self._build_consolidated(row, available_candidate), axis=1
        )
        self.df["job_consolidated"] = self.df.apply(
            lambda row: self._build_consolidated(row, available_job), axis=1
        )
        return self.df

    @staticmethod
    def _clean_str(s):
        s = s.replace('\xa0', ' ').replace('•', '').replace('·', '')
        s = re.sub(r' +', ' ', s).strip()
        return s

    def clean_and_remove_columns(self):
        for column in self.df.columns:
            if self.df[column].dtype == 'object' or self.df[column].dtype == 'str':
                self.df[column] = self.df[column].apply(
                    lambda x: next(
                        ([self._clean_str(s) for s in x.split(d) if self._clean_str(s)]
                         for d in self.delimiters if d in x),
                        x
                    ) if isinstance(x, str) and pd.notna(x) else x
                )

        for feature in self.df.columns:
            self.df[feature] = self.df[feature].apply(self._clean_and_check)

        for column in self.df.columns:
            if self.df[column].dtype == 'str' or self.df[column].dtype == 'object':
                self.df[column] = self.df[column].apply(self._to_list_or_fill)

        self.df.dropna(axis=1, thresh=self.column_removal_threshold, inplace=True)
        return self.df

    @staticmethod
    def _to_list_or_fill(x):
        if isinstance(x, list):
            if len(x) == 0:
                return np.nan
            else:
                return x
        else:
            return x

    def _clean_and_check(self, x):
        if isinstance(x, str):
            x = x.strip()
            if x == '':
                return np.nan
            try:
                x = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x

        parsed = x

        if not isinstance(parsed, list):
            return parsed

        if parsed and all(isinstance(i, list) for i in parsed):
            flat = [item for sublist in parsed for item in sublist]
        else:
            flat = parsed

        cleaned = []
        for v in flat:
            if isinstance(v, (list, np.ndarray)):
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(v, str):
                v = v.strip()
            if v in self.bad_values:
                continue
            cleaned.append(v)

        return cleaned if cleaned else np.nan

    def impute(self):
        for column in self.df.columns:
            sample = self.df[column].dropna()
            if not sample.empty and isinstance(sample.iloc[0], list):
                self.df[column] = self.df[column].apply(
                    lambda x: ['unknown'] if not isinstance(x, list) else x
                )
            elif self.df[column].dtype == 'str':
                self.df[column] = self.df[column].apply(
                    lambda x: 'unknown' if pd.isna(x) else x
                )
            elif self.df[column].dtype != 'float64':
                self.df[column] = self.df[column].apply(
                    lambda x: x[0] if isinstance(x, list) else 'unknown' if pd.isna(x) else x
                )
        return self.df