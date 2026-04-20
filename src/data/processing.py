import os

import numpy as np
import pandas as pd

from src.utils.logger import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class PreprocessingCICIDS2017:
    def __init__(self, path_raw_dataset: str, classes: str = "category", apply_undersampling: bool = False):
        self.path_raw_dataset = path_raw_dataset
        self.classes = classes
        self.apply_undersampling = apply_undersampling
        self.df = self.preprocess()

    def load_csvs(self):
        all_files = []
        for file in os.listdir(self.path_raw_dataset):
            if file.endswith(".csv"):
                all_files.append(os.path.join(self.path_raw_dataset, file))
        df = pd.concat((pd.read_csv(f, encoding="latin1") for f in all_files), ignore_index=True)
        df.columns = df.columns.str.strip()
        return df

    def undersample_benign_batches(self, df, batch_size=200, benign_threshold=1.0, benign_keep_prob=0.1):
        logger.debug("Applying benign undersampling")
        df = df.sort_values("Timestamp").reset_index(drop=True)
        kept_rows = []
        n_batches = (len(df) + batch_size - 1) // batch_size

        for i in range(n_batches):
            batch = df.iloc[i * batch_size : (i + 1) * batch_size]
            labels = batch["Label"]
            benign_ratio = (labels == "BENIGN").mean()

            if benign_ratio >= benign_threshold:
                if np.random.rand() < benign_keep_prob:
                    kept_rows.append(batch)
            else:
                kept_rows.append(batch)

        new_df = pd.concat(kept_rows).reset_index(drop=True)
        logger.debug(f"After undersampling: {len(new_df)} flows remaining (out of {len(df)})")
        return new_df

    def encode(self, dataframe):
        logger.debug("Encoding features")
        dataframe = dataframe.copy()
        n_rows_before = dataframe.shape[0]
        dataframe.dropna(inplace=True, how="all", axis=1)
        n_rows_after = dataframe.shape[0]
        logger.debug(f"Removed {n_rows_before - n_rows_after} rows with all NaN values")

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        n_rows_before = dataframe.shape[0]
        dataframe.dropna(inplace=True, how="any", axis=0)
        n_rows_after = dataframe.shape[0]
        logger.debug(f"Removed {n_rows_before - n_rows_after} rows with NaN values")

        if "Source Port" in dataframe.columns and "Destination Port" in dataframe.columns:
            dataframe["Source Port"] = pd.to_numeric(dataframe["Source Port"], errors="coerce")
            dataframe["Destination Port"] = pd.to_numeric(dataframe["Destination Port"], errors="coerce")
        elif "Src Port" in dataframe.columns and "Dst Port" in dataframe.columns:
            dataframe["Src Port"] = pd.to_numeric(dataframe["Src Port"], errors="coerce")
            dataframe["Dst Port"] = pd.to_numeric(dataframe["Dst Port"], errors="coerce")
            dataframe.rename(columns={"Src Port": "Source Port", "Dst Port": "Destination Port"}, inplace=True)

        if "Src IP" in dataframe.columns and "Dst IP" in dataframe.columns:
            dataframe.rename(columns={"Src IP": "Source IP", "Dst IP": "Destination IP"}, inplace=True)

        dataframe = pd.get_dummies(dataframe, columns=["Protocol"])

        if self.classes == "category":
            dataframe = pd.get_dummies(dataframe, columns=["Label"], prefix="Attack", sparse=True)
        elif self.classes == "binary":
            dataframe["Attack"] = dataframe["Label"]
            dataframe["Label"] = dataframe["Label"].apply(lambda x: 1 if x != "BENIGN" else 0)

        return dataframe

    def preprocess(self):
        df = self.load_csvs()
        if self.apply_undersampling:
            logger.debug("Applying undersampling")
            df = self.undersample_benign_batches(df)
        return self.encode(df)


class PreprocessingXIIOTID:
    def __init__(self, path_raw_dataset: str, classes: str = "category"):
        self.path_raw_dataset = path_raw_dataset
        self.classes = classes
        self.df = self.preprocess()

    def load_csvs(self) -> pd.DataFrame:
        df_path = os.path.join(self.path_raw_dataset, "X-IIoTID.csv")
        return pd.read_csv(df_path)

    def encode(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Encoding features")
        dataframe.dropna(inplace=True, how="all", axis=1)

        columns_to_drop = [
            "Date",
            "Avg_user_time",
            "Std_user_time",
            "Avg_nice_time",
            "Std_nice_time",
            "Avg_system_time",
            "Std_system_time",
            "Avg_IO_wait_time",
            "Std_IO_wait_time",
            "Avg_idle_time",
            "Std_idle_time",
            "Avg_tps",
            "Std_tps",
            "Avg_rtps",
            "Std_rtps",
            "Avg_wtps",
            "Std_wtps",
            "Avg_ldavg_1",
            "Std_ldavg_1",
            "Avg_Kbmemused",
            "Std_Kbmemused",
            "Avg_num_proc/s",
            "Std_num_proc/s",
            "Avg_num_swch/s",
            "Std_num_swch/s",
            "read_write_physical.process",
            "Avg_num_cswch/s",
            "Anomaly_Alert",
            "OSSEC_alert",
            "Alert_level",
            "R_W_physicial",
            "File_act",
            "Proc_act",
            "Is_privileged",
            "Login_attmp",
            "Succ_login",
            "anomaly_alert",
            "Avg_ideal_time",
            "std_num_cswch/s",
            "Avg_num_Proc/s",
            "Std_kbmemused",
            "Avg_kbmemused",
            "Std_iowait_time",
            "Avg_iowait_time",
            "Process_activity",
            "File_activity",
            "Login_attempt",
            "Succesful_login",
            "is_privileged",
            "OSSEC_alert_level",
            "Std_ideal_time",
        ]

        columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]
        dataframe.drop(columns=columns_to_drop, axis=1, inplace=True)

        dataframe["Timestamp"] = pd.to_numeric(dataframe["Timestamp"], errors="coerce")
        dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"], unit="s", errors="coerce")

        n_rows_before = dataframe.shape[0]
        dataframe.dropna(subset=["Timestamp"], inplace=True)
        n_rows_after = dataframe.shape[0]
        logger.info(f"Dropped {n_rows_before - n_rows_after} rows with invalid or missing 'Timestamp' values.")

        dataframe.sort_values(by=["Timestamp"], inplace=True)

        if self.classes == "category":
            dataframe.drop(["class3", "class1"], axis=1, inplace=True)
            dataframe = pd.get_dummies(dataframe, columns=["Protocol", "Service", "class2"])
        elif self.classes == "binary":
            dataframe["Attack"] = dataframe["class1"]
            dataframe.drop(["class1"], axis=1, inplace=True)
            dataframe.drop(["class2"], axis=1, inplace=True)
            dataframe = pd.get_dummies(dataframe, columns=["Protocol", "Service"])
            dataframe["class3"] = dataframe["class3"].apply(lambda x: 0 if x == "Normal" else 1)

        return dataframe

    def preprocess(self) -> pd.DataFrame:
        df = self.load_csvs()
        return self.encode(df)
