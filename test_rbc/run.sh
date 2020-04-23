
# Set custom mpi command if needed (e.g. with core count)
mpi_command="mpiexec"

# Loop over test directories
for testdir in ./test*/
do
    cd $testdir
    $mpi_command python3 ../simulation.py
    python3 ../plot_memory.py memory.txt --output=memory_$(basename $testdir).pdf
    cd ..
done

# Consolidate plots
rm -rf plots
mkdir plots
cp -f test*/memory*.pdf plots
