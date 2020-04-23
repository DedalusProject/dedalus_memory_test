
alias mpi_command="mpiexec"

for testdir in ./test*/
do
    cd $testdir
    mpi_command python3 ../simulation.py
    python3 ../plot_memory.py memory.txt --output=memory_$(basename $testdir).pdf
    cd ..
done

rm -rf plots
mkdir plots
cp -f test*/memory*.pdf plots
