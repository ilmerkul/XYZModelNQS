figure-10-energy:
	gnuplot -e "filename='project/data/report_10.csv'" project/gnuplot/energy_and_magnetization.gp > publication/figures/10/energy_and_magnetizaton.png
figure-21-energy:
	gnuplot -e "filename='project/data/report_21.csv'" project/gnuplot/energy_and_magnetization.gp > publication/figures/21/energy_and_magnetizaton.png
figure-21-2-energy:
	gnuplot -e "filename='project/data/report_21_xx_lam_1.0.csv'" project/gnuplot/energy_and_magnetization.gp > publication/figures/21-2/energy_and_magnetization.png
figure-32-energy:
	gnuplot -e "filename='project/data/report_32.csv'" project/gnuplot/energy_and_magnetization.gp > publication/figures/32/energy_and_magnetizaton.png
figure-64-energy:
	gnuplot -e "filename='project/data/report_64.csv'" project/gnuplot/energy_and_magnetization.gp > publication/figures/64/energy_and_magnetizaton.png

figure-10-correlations:
	gnuplot -e "filename='project/data/report_10.csv'" project/gnuplot/energy_and_zz.gp > publication/figures/10/energy_and_zz.png
figure-21-correlations:
	gnuplot -e "filename='project/data/report_21.csv'" project/gnuplot/energy_and_zz.gp > publication/figures/21/energy_and_zz.png
figure-21-2-correlations:
	gnuplot -e "filename='project/data/report_21_xx_lam_1.0.csv'" project/gnuplot/energy_and_zz.gp > publication/figures/21-2/energy_and_zz.png
figure-32-correlations:
	gnuplot -e "filename='project/data/report_32.csv'" project/gnuplot/energy_and_zz.gp > publication/figures/32/energy_and_zz.png
figure-64-correlations:
	gnuplot -e "filename='project/data/report_64.csv'" project/gnuplot/energy_and_zz.gp > publication/figures/64/energy_and_zz.png

figures: figure-10-energy figure-21-energy figure-32-energy figure-64-energy figure-10-correlations figure-21-correlations figure-32-correlations figure-64-correlations figure-21-2-energy figure-21-2-correlations

install:
	poetry lock
	poetry install

install-gpu:
	poetry lock
	poetry install
	poetry run pip install --upgrade "jax[cuda110]"==0.2.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html
