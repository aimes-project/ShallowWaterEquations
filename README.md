# ShallowWaterEquations
GGDML code to solve shallow water equations

To translate source code into C code, please run the compile.sh script.
The folder includes example configuration files (.conf) that lead to generate code for serial execution, OpenMP annotated code, and apply cache blocking.

The tool is called by the compile.sh script.
It is provided as a single .zip file.
Python 2.7 is needed to run the tool.
The source code of the tool will be published later.
When run, the compile.sh script generates three folder, which are clones of the src directory with different transformations applied to the code. 
