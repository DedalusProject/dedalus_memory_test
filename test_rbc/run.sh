
cd test1
mpiexec python3 ../simulation.py
python3 ../plot_memory.py memory.txt --output=memory1.pdf

cd ../test2
mpiexec python3 ../simulation.py
python3 ../plot_memory.py memory.txt --output=memory2.pdf

cd ../test3
mpiexec python3 ../simulation.py
python3 ../plot_memory.py memory.txt --output=memory3.pdf

cd ..
\cp -f */memory*.pdf .
