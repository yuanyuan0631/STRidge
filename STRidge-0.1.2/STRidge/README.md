# STRidge



## Introduction
This project provides a straightforward and efficient method to derive specific forms of partial differential equations from real data. By calling the STRidge function, you can implement its core algorithm, significantly simplifying the process of model building.



## Installation
Here is an example installation process using the Anaconda3 environment:

1. Download the compressed package from the database and extract it.
2. Locate the directory where Python packages are installed via pip/conda (usually the installation path of Anaconda: `/lib/site-packages`). Copy the extracted files to this directory, then open the "STRidge" folder and copy its path.
3. Open a command prompt, navigate to the path copied in the previous step using `cd`, and run the command `pip install .`. Wait for the "installation successful" message (this takes about 20-30 seconds. If you get an access denied message, rerun the command prompt as an administrator and repeat the steps).
4. To verify the installation, enter the command `python` to switch to the Python environment, and run `import STRidge`. If there are no errors, the installation was successful!



## Usage and Parameters
To use the package, simply import it with `import STRidge`, and then instantiate it directly (e.g., `a = STRidge(R0, y, lambda, train_it)`).

Below is a detailed explanation of the parameters in the STRidge module:

### STRidge Function Definition
```python
STRidge(R0, Ut, lambda1, train_it, lam=1e-5, d_tol=1, maxit=100,
        STR_iters=10, l0_penalty=None, normalize=2, split=0.8,
        print_best_tol=False, tol=None, l0_penalty_0=None)
```

- `R0`: Candidate terms for the partial differential equation (can be in forms like `torch.tensor` or `np.array`)
- `y`: Target values (can be in forms like `torch.tensor` or `np.array`)
- `lambda1`: Neural network parameter input for determining the index vector of `R0`
- `train_it`: Number of optimization iterations for the main training function when calling the STRidge method (if `train_it` is not 0, `l0_penalty_0` and `tol` parameters must be provided! A warning will be issued if these parameters are not specified, and the program will fail if they are missing)
- `lam` (default `1e-5`): Regularization parameter to control model complexity and prevent overfitting
- `d_tol` (default `1`): Initial tolerance
- `maxit` (default `100`): Maximum number of iterations for the optimization process
- `STR_iters` (default `10`): Maximum number of iterations for a single call of STRidge
- `l0_penalty` (default `None`): Initial l0 regularization penalty term
- `normalize` (default `2`): Normalization coefficient
- `split` (default `0.8`): Ratio for splitting training and validation sets
- `print_best_tol` (default `False`): Whether to print the best tolerance and STRidge training error
- `tol` (default `None`): Tolerance from the previous round of PINN training; must be provided if `train_it` is not 0!
- `l0_penalty_0` (default `None`): l0 regularization penalty term from the previous round of PINN training; must be provided if `train_it` is not 0!



## Contact
The code is owned by the author yuanyuan0631. If you encounter any issues during installation or usage, please contact the author at yuanyuan0631@gmail.com.

