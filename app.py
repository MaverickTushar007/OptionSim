import streamlit as st
import numpy as np
import scipy.linalg as sla
import pandas as pd


def LU_decom(A):
    temp = np.array(A, dtype=float)
    P, L, U = sla.lu(temp)
    return P, L, U


def EVD_decom(A):
    temp = np.array(A, dtype=float)
    w, v = np.linalg.eig(temp)
    return w, v


def SVD_decom(A):
    temp = np.array(A, dtype=float)
    U, S, VT = np.linalg.svd(temp)
    return U, S, VT


def display_matrix(matrix, name):
    st.write(f"**Matrix {name}:**")
    real = np.real(matrix) if np.iscomplexobj(matrix) else matrix
    st.dataframe(pd.DataFrame(np.round(real, 4)), use_container_width=False)


st.set_page_config(page_title="Matrix Decomposition Visualizer", layout="wide")
st.title("Matrix Decomposition Visualizer")
st.markdown("Enter a matrix, choose a decomposition method, and view the results.")

st.sidebar.header("1. Matrix Input Section")

rows = st.sidebar.number_input("Number of Rows", min_value=1, max_value=5, value=3, step=1)
cols = st.sidebar.number_input("Number of Columns", min_value=1, max_value=5, value=3, step=1)

st.sidebar.markdown("### Enter Matrix Values")

default_df = pd.DataFrame(np.zeros((rows, cols)))

edited_df = st.sidebar.data_editor(
    default_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True
)

A = edited_df.to_numpy(dtype=float)

st.sidebar.header("2. Select Decomposition")
method = st.sidebar.selectbox(
    "Choose a method:",
    ["LU Decomposition", "QR Decomposition", "Eigenvalue Decomposition", "Singular Value Decomposition (SVD)", "Cholesky Decomposition"]
)

st.header("Results")
st.subheader("Original Matrix (A)")
st.dataframe(pd.DataFrame(A))

if st.sidebar.button("Compute Decomposition"):
    try:
        if method == "LU Decomposition":
            st.markdown("### LU Decomposition ($A = P L U$)")
            P, L, U = LU_decom(A)
            col1, col2, col3 = st.columns(3)
            with col1:
                display_matrix(P, "P (Permutation)")
            with col2:
                display_matrix(L, "L (Lower Triangular)")
            with col3:
                display_matrix(U, "U (Upper Triangular)")

        elif method == "QR Decomposition":
            st.markdown("### QR Decomposition ($A = Q R$)")
            Q, R = np.linalg.qr(A)
            col1, col2 = st.columns(2)
            with col1:
                display_matrix(Q, "Q (Orthogonal)")
            with col2:
                display_matrix(R, "R (Upper Triangular)")

        elif method == "Eigenvalue Decomposition":
            st.markdown("### Eigenvalue Decomposition ($A V = V \\Lambda$)")
            if rows != cols:
                st.error("Eigenvalue decomposition requires a square matrix.")
            else:
                eigenvalues, eigenvectors = EVD_decom(A)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Eigenvalues:**")
                    eig_df = pd.DataFrame({
                        "Real": np.round(np.real(eigenvalues), 4),
                        "Imaginary": np.round(np.imag(eigenvalues), 4)
                    })
                    st.dataframe(eig_df)
                with col2:
                    display_matrix(eigenvectors, "V (Eigenvectors)")

        elif method == "Singular Value Decomposition (SVD)":
            st.markdown("### Singular Value Decomposition ($A = U \\Sigma V^T$)")
            U, S, VT = SVD_decom(A)
            Sigma = np.zeros((A.shape[0], A.shape[1]))
            np.fill_diagonal(Sigma, S[:min(A.shape)])
            col1, col2, col3 = st.columns(3)
            with col1:
                display_matrix(U, "U")
            with col2:
                display_matrix(Sigma, "Σ (Singular Values)")
            with col3:
                display_matrix(VT, "V^T")

        elif method == "Cholesky Decomposition":
            st.markdown("### Cholesky Decomposition ($A = L L^T$)")
            if rows != cols:
                st.error("Cholesky decomposition requires a square matrix.")
            elif not np.allclose(A, A.T, atol=1e-8):
                st.error("Cholesky decomposition requires a symmetric matrix.")
            elif np.any(np.linalg.eigvalsh(A) <= 0):
                st.error("Cholesky decomposition requires a positive-definite matrix.")
            else:
                L = np.linalg.cholesky(A)
                col1, col2 = st.columns(2)
                with col1:
                    display_matrix(L, "L (Lower Triangular)")
                with col2:
                    display_matrix(L.T, "L^T (Upper Triangular)")

    except np.linalg.LinAlgError as e:
        st.error(f"Linear Algebra Error: {e}")
        st.info("Make sure the matrix meets the mathematical requirements for the chosen decomposition (e.g., positive definite for Cholesky).")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")