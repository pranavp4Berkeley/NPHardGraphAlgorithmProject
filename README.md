# Attempted Solution to the TA dropoff problem

- You have all of the input files for the project in the folder "inputs" (The input folder from this submission is a little out of place (since we are running different sections of the inputs on different local computers) so you are going to have to use your own input folder with all of the inputs you want to test inside.
- You want to save the files outputted by the solver in the folder "outputs"
- You need to first install networkx, matplotlib, and numpy with: pip3 install networkx matplotlib numpy
- You can just run the solver on all the files by doing: python3 solver.py --all ./inputs
- For all the inputs of size 50, we used a different algorithm by utilizing the idea of simulated annealing, which seems to be a common idea when thinking about TSP. 
- For those inputs, what you can do is move the rest of the inputs (all of the _100.in and all the _200.in inputs) out of the folder (and into another) and then use the same command python3 solver.py --all ./inputs on the remaining inputs in that folder (all of the _50.in) after you uncomment the lines 193 to 203 and comment out the line 204.