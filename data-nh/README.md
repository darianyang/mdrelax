### NH NMR Relaxation Data for T4L

The data here was kindly shared by Lewis Kay and Eva Meirovitch and corresponds to the following papers: Meirovitch 2012 JPC-B and Goto et al. 2001 JMB 

* T4l_Good_500_800_25.dat : This is a txt file which contains 298 K, 11.7 and 18.8 T data. The six numbers are T1, T1(error), T2, T2(error), NOE, NOE(error) for 11.7 T (500MHz) followed in the same format for 18.8 T (800MHz).
    * Likely 15N labeled WT* T4L

* nt1t2noe.t4 and R1R2NOE.in: Relaxation data at 600MHz, the r1r2noe input is for quadric, the results of which are as follows:
    * 15N/13C labeled WT T4L

```
          ****************** Axial Results *******************


 Parameters          Actual
Diso (10^-7/s)     1.59025 +/-      0.00103
Dpar/Dper          1.32419 +/-      0.00482
Theta              0.67270 +/-      0.00795
Phi                0.86404 +/-      0.01140

 Parameters          Jacknife
Diso (10^-7/s)     1.59081 +/-      0.00516
Dpar/Dper          1.32249 +/-      0.02770
Theta              0.67714 +/-      0.05578
Phi                0.88159 +/-      0.07219

 X2    2491.864         X2 (red)    21.85846    

 F      88.45821         #points           118
```
