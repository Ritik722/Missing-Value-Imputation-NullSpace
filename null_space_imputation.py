import os
import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame | None:
    """
    Load the dataset. Returns None if file is missing.
    Expected: a CSV with numeric feature columns (e.g., x1..x5).
    """
    if not os.path.exists(csv_path):
        print(f"[INFO] File not found: {csv_path}")
        print("[INFO] This repository does not include data. Please add your own CSV file.")
        print("[INFO] Example: place 'GTPvar.csv' in the same folder as this script.")
        return None

    return pd.read_csv(csv_path, index_col=0)


def add_missing_count_column(df: pd.DataFrame, col_name: str = "NApresent") -> pd.DataFrame:
    """
    Adds a column that counts missing values per row.
    """
    df = df.copy()
    df[col_name] = df.isnull().sum(axis=1)
    return df


def get_complete_rows_matrix(df: pd.DataFrame, missing_col: str = "NApresent") -> np.ndarray:
    """
    Extracts fully complete rows (no missing values) and returns them as a NumPy matrix.
    """
    complete = df[df[missing_col] == 0].drop(columns=[missing_col])
    return complete.to_numpy(dtype=float)


def compute_nullspace_constraints(data_matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Computes null space constraints using SVD.

    We find a basis for the null space of (data_matrix^T).
    If singular values are below tolerance, we treat them as ~0.
    The constraint matrix A is returned in the form:
        A x = 0
    where each row of A is one linear constraint.
    """
    # Keep your orientation: SVD on transpose
    V, S, U = np.linalg.svd(data_matrix.T)

    # Your original rank estimation approach
    # new_rank = min(shape) - number_of_singular_values_below_tol
    new_rank = min(data_matrix.shape) - np.abs(S)[::-1].searchsorted(tol)

    # Null space basis vectors (your A)
    A = V[:, new_rank:]
    A = A.T  # constraints as rows
    return A


def impute_using_nullspace_equations(df: pd.DataFrame, A: np.ndarray, missing_col: str = "NApresent") -> pd.DataFrame:
    """
    Impute missing values row-wise using null space equations.

    Your original logic:
    - For each row i, choose 'equation_req' = df[NApresent] for that row
    - Split indices into missing (aID) and known (bID)
    - Build linear system:
        a * x = b
      where
        a = A[:equation_req, aID]
        b = -A[:equation_req, bID] @ known_values
    - Solve for x and fill missing positions
    """
    df = df.copy()
    len_A = len(A)
    total_rows = len(df)

    for i in range(total_rows):
        equation_req = df.iloc[i][missing_col]

        # Skip invalid requests (same intent as your original guard)
        # Using python 'or' (safe + clear)
        if equation_req == 0 or equation_req > len_A:
            continue

        aID = np.empty(0, dtype="int64")  # missing indices
        bID = np.empty(0, dtype="int64")  # known indices

        # iterate feature columns only (exclude NApresent)
        for j in range(len(df.columns) - 1):
            if pd.isnull(df.iloc[i, j]):
                aID = np.append(aID, j)
            else:
                bID = np.append(bID, j)

        # If nothing is missing, nothing to solve
        if len(aID) == 0:
            continue

        # Build system from the first 'equation_req' constraints
        a = A[:equation_req, aID]
        b2 = -A[:equation_req, bID]
        known_vals = df.iloc[i, bID].to_numpy(dtype=float)
        b = np.dot(b2, known_vals)

        # Solve only if system is square; otherwise skip (keeps it safe)
        if a.shape[0] == a.shape[1]:
            x = np.linalg.solve(a, b)
            df.iloc[i, aID] = x
        else:
            # Not enough / too many equations for the unknowns in this row
            # (Keeping your logic: do not change behavior to lstsq automatically)
            continue

    return df


def main():
    CSV_PATH = "GTPvar.csv"

    df = load_data(CSV_PATH)
    if df is None:
        return  # clean exit, no error

    # Add missing count column
    df = add_missing_count_column(df, col_name="NApresent")

    # Create complete-data matrix for SVD
    data_matrix = get_complete_rows_matrix(df, missing_col="NApresent")
    if data_matrix.size == 0:
        print("[INFO] No complete rows found. Cannot compute null space constraints.")
        return

    # Compute A (null space constraint matrix)
    A = compute_nullspace_constraints(data_matrix, tol=1e-8)

    # Impute missing values using your equation selection logic
    df_imputed = impute_using_nullspace_equations(df, A, missing_col="NApresent")

    # Drop helper column and save
    df_imputed = df_imputed.drop(columns=["NApresent"])
    df_imputed.to_csv("GTPvar_imputed.csv")
    print("[DONE] Saved: GTPvar_imputed.csv")


if __name__ == "__main__":
    main()
