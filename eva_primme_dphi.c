/*******************************************************************************
 *
 * File eva_primme_dphi.c
 *
 * This software is distributed under the terms of the GNU General Public
 * License (GPL)
 *
 *
 * Syntax: eva_primme -i <input file>
 *
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "forces.h"
#include "version.h"
#include "global.h"
#include "linalg.h"
#include "update.h"
#define ABS(a) ((a) < 0 ? -(a) : (a))

#if (defined _OPENMP)
#include <omp.h>
#endif

#include "primme.h" /* header file is required to run primme */

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

void MatMult_Dw_primme(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr);
void MatMult_Dw_primme_Preconditioner(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme,
                                      int *ierr);

static void par_GlobalSum(void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr);
static void broadcastForDouble(void *buffer, int *count, primme_params *primme, int *ierr);
int mpierr;

typedef enum { EVA_QHAT, EVA_DW } evaop_t;

typedef enum { EVA_SMALL, EVA_LARGE, EVA_ALL } evatarget_t;

static struct {
    int nev, target, opid;
    double tol;
    char opname[16];
} evadat;

typedef union {
    spinor_dble s;
    complex_dble r[12];
} spin_dble_t;

static int my_rank, endian, append;
static int first, last, step;
static int ifail[2], is;
static int ipgrd[3];

static iodat_t iodat[1];
static char nbase[NAME_SIZE], log_dir[NAME_SIZE];
static char log_file[NAME_SIZE], log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE], end_file[NAME_SIZE];
static FILE *fin = NULL, *flog = NULL, *fend = NULL;

static void read_dirs(void) {
    if (my_rank == 0) {
        find_section("Run name");
        read_line("name", "%s", nbase);

        find_section("Log and data directories");
        read_line("log_dir", "%s", log_dir);
    }

    MPI_Bcast(nbase, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(log_dir, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
}

static void setup_files(void) {
    error(name_size("%s/%s.eva_primme_dphi.log~", log_dir, nbase) >= NAME_SIZE, 1, "setup_files [eva_primme_dphi.c]",
          "log_dir name is too long");

    sprintf(log_file, "%s/%s.eva_primme_dphi.log", log_dir, nbase);
    sprintf(end_file, "%s/%s.eva_primme_dphi.end", log_dir, nbase);
    sprintf(log_save, "%s~", log_file);

    check_dir_root(log_dir);
}

static void read_cnfg_range(void) {
    if (my_rank == 0) {
        find_section("Configurations");

        read_line("first", "%d", &first);
        read_line("last", "%d", &last);
        read_line("step", "%d", &step);

        error_root((first < 1) || (last < first) || (step < 1) || ((last - first) % step != 0), 1,
                   "read_cnfg_range [eva_primme_dphi.c]", "Improper configuration range");
    }

    MPI_Bcast(&first, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&last, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&step, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

static void read_primme_parms(void) {
    int opid, nev, targ;
    double tol;
    char name[NAME_SIZE], targname[NAME_SIZE];

    if (my_rank == 0) {
        find_section("Eigenvalues");
        read_line("nev", "%d", &nev);
        read_line("tolerance", "%lf", &tol);

        error_root((nev < 1) || (tol < 0.0), 1, "read_primme_parms [eva_primme_dphi.c]", "Parameters are out of range");

        read_line("operator", "%s", name);
        read_line("target", "%s", targname);

        if (strcmp(name, "Qhat") == 0) {
            opid = EVA_QHAT;
        } else if (strcmp(name, "Dw") == 0) {
            opid = EVA_DW;
        } else {
            error_root(1, 1, "read_primme_parms [eva_primme_dphi.c]", "Unknown matrix type");
        }

        if (strcmp(targname, "small") == 0) {
            targ = primme_closest_abs;
        } else if (strcmp(targname, "large") == 0) {
            targ = primme_largest_abs;
        } else {
            error_root(1, 1, "read_primme_parms [eva_primme_dphi.c]", "Unknown eigenvalue search target");
        }
    }

    mpierr = MPI_Bcast(&opid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpierr = MPI_Bcast(&targ, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpierr = MPI_Bcast(&nev, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mpierr = MPI_Bcast(&name, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    mpierr = MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    evadat.opid = opid;
    evadat.target = targ;
    evadat.nev = nev;
    evadat.tol = tol;
    strcpy(evadat.opname, name);
}

static void read_infile(int argc, char *argv[]) {
    int ifile;

    if (my_rank == 0) {
        flog = freopen("STARTUP_ERROR", "w", stdout);

        ifile = find_opt(argc, argv, "-i");
        endian = endianness();

        error_root((ifile == 0) || (ifile == (argc - 1)), 1, "read_infile [eva_primme_dphi.c]",
                   "Syntax: eva_primme -i <input file>");

        error_root(endian == UNKNOWN_ENDIAN, 1, "read_infile [eva_primme_dphi.c]", "Machine has unknown endianness");

        fin = freopen(argv[ifile + 1], "r", stdin);
        error_root(fin == NULL, 1, "read_infile [eva_primme_dphi.c]", "Unable to open input file");

        append = find_opt(argc, argv, "-a");
    }

    MPI_Bcast(&endian, 1, MPI_INT, 0, MPI_COMM_WORLD);

    read_dirs();
    read_iodat("Configurations", "i", iodat);
    read_cnfg_range();
    read_bc_parms("Boundary conditions", 0x2);

    read_sap_parms("SAP", 0x1);

    read_dfl_parms("Deflation subspace");
    read_dfl_pro_parms("Deflation projection");
    read_dfl_gen_parms("Deflation subspace generation");
    read_lat_parms("Lattice parameters", 0x2);

    read_primme_parms();
    setup_files();

    if (my_rank == 0) { fclose(fin); }
}

static void check_old_log(int *fst, int *lst, int *stp) {
    int ie, ic, isv;
    int fc, lc, dc, pc;
    int nt, np[4], bp[4];
    char line[NAME_SIZE];

    fend = fopen(log_file, "r");
    error_root(fend == NULL, 1, "check_old_log [eva_primme_dphi.c]", "Unable to open log file");

    fc = 0;
    lc = 0;
    dc = 0;
    pc = 0;

    ie = 0x0;
    ic = 0;
    isv = 0;

    while (fgets(line, NAME_SIZE, fend) != NULL) {
        if ((strstr(line, "MPI process grid") != NULL) && (strstr(line, "changed") == NULL)) {
            if (sscanf(line, "%dx%dx%dx%d MPI process grid, %dx%dx%dx%d", np, np + 1, np + 2, np + 3, bp, bp + 1, bp + 2,
                       bp + 3) == 8) {
                ipgrd[0] = ((np[0] != NPROC0) || (np[1] != NPROC1) || (np[2] != NPROC2) || (np[3] != NPROC3));
                ipgrd[1] = ((bp[0] != NPROC0_BLK) || (bp[1] != NPROC1_BLK) || (bp[2] != NPROC2_BLK) || (bp[3] != NPROC3_BLK));
            } else {
                ie |= 0x1;
            }
        } else if ((strstr(line, "OpenMP thread") != NULL) && (strstr(line, "changed") == NULL)) {
            if (sscanf(line, "%d OpenMP thread", &nt) == 1) {
                ipgrd[2] = (nt != NTHREAD);
            } else {
                ie |= 0x1;
            }
        } else if (strstr(line, "fully processed") != NULL) {
            pc = lc;

            if (sscanf(line, "Configuration no %d", &lc) == 1) {
                ic += 1;
                isv = 1;
            } else {
                ie |= 0x1;
            }

            if (ic == 1) {
                fc = lc;
            } else if (ic == 2) {
                dc = lc - fc;
            } else if ((ic > 2) && (lc != (pc + dc))) {
                ie |= 0x2;
            }
        } else if (strstr(line, "Configuration no") != NULL) {
            isv = 0;
        }
    }

    fclose(fend);

    error_root((ie & 0x1) != 0x0, 1, "check_old_log [eva_primme_dphi.c]", "Incorrect read count");
    error_root((ie & 0x2) != 0x0, 1, "check_old_log [eva_primme_dphi.c]", "Configuration numbers are not equally spaced");
    error_root(isv == 0, 1, "check_old_log [eva_primme_dphi.c]", "Log file extends beyond the last configuration save");

    (*fst) = fc;
    (*lst) = lc;
    (*stp) = dc;
}

static void check_files(void) {
    int ie;
    int fst, lst, stp;

    ipgrd[0] = 0;
    ipgrd[1] = 0;
    ipgrd[2] = 0;

    if (my_rank == 0) {
        if (append) {
            check_old_log(&fst, &lst, &stp);

            error_root((fst != lst) && (stp != step), 1, "check_files [eva_primme_dphi.c]",
                       "Continuation run:\n"
                       "Previous run had a different configuration separation");
            error_root(first != lst + step, 1, "check_files [eva_primme_dphi.c]",
                       "Continuation run:\n"
                       "Configuration range does not continue the previous one");
        } else {
            ie = check_file(log_file, "r");

            error_root(ie != 0, 1, "check_files [eva_primme_dphi.c]", "Attempt to overwrite old *.log  file");
        }
    }

    error(name_size("%sn%d", nbase, last) >= NAME_SIZE, 1, "check_files [eva_primme_dphi.c]",
          "Configuration base name is too long");
    sprintf(cnfg_file, "%sn%d", nbase, last);
    check_iodat(iodat, "i", 0x1, cnfg_file);
}

static void print_info(void) {
    long ip;

    if (my_rank == 0) {
        ip = ftell(flog);
        fclose(flog);

        if (ip == 0L) { remove("STARTUP_ERROR"); }

        if (append) {
            flog = freopen(log_file, "a", stdout);
        } else {
            flog = freopen(log_file, "w", stdout);
        }
        error_root(flog == NULL, 1, "print_info [eva_primme_dphi.c]", "Unable to open log file");
        if (append) {
            printf("Continuation run\n\n");
        } else {
            printf("\nEigenvalue spectrum calculation of selected Dirac operator with Primme\n");
            printf("----------------------------------------------------------------------\n\n");

            printf("Program version %s\n", openQCD_RELEASE);

            if (endian == LITTLE_ENDIAN) {
                printf("The machine is little endian\n");
            } else {
                printf("The machine is big endian\n");
            }

            print_lattice_sizes();
            print_lat_parms(0x2);
            print_bc_parms(0x2);

            print_sap_parms(0x0);

            print_dfl_parms(0x0);

            print_iodat("i", iodat);
            printf("Configurations no %d -> %d in steps of %d\n\n", first, last, step);
        }
        fflush(flog);
    }
}

static void maxn(int *n, int m) {
    if ((*n) < m) { (*n) = m; }
}

static void dfl_wsize(int *nws, int *nwv, int *nwvd) {
    dfl_parms_t dp;
    dfl_pro_parms_t dpr;

    dp = dfl_parms();
    dpr = dfl_pro_parms();
    maxn(nws, dp.Ns + 2 * dpr.nkv + 14);
    maxn(nwv, 2 * dpr.nmx_gcr + 3);
    maxn(nwvd, 2 * dpr.nkv + 4);
}

static void wsize(int *nws, int *nwv, int *nwvd) {
    (*nws) = 0;
    (*nwv) = 0;
    (*nwvd) = 0;

    dfl_wsize(nws, nwv, nwvd);
}

static void check_endflag(int *iend) {
    if (my_rank == 0) {
        fend = fopen(end_file, "r");

        if (fend != NULL) {
            fclose(fend);
            remove(end_file);
            (*iend) = 1;
            printf("End flag set, run stopped\n\n");
        } else {
            (*iend) = 0;
        }
    }

    MPI_Bcast(iend, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    int nc, iend, *status, ret, i, j;
    int nws, nwv, nwvd;
    qflt qr;
    int nsites;
    int lastrun;
    double wt1, wt2, wtavg;
    spinor_dble **wscheck;
    complex_qflt dlambda;
    qflt rqsm;

    double del, w1, *w2, starteval, maxm0;

    double m0; /*bare mass*/
    /* PRIMME configuration struct */

    double *evals; /* Array with the computed eigenvalues */
    double *rnorms; /* Array with the computed eigenpairs residual norms */
    PRIMME_COMPLEX_DOUBLE *evecs = NULL; /* Array with the computed eigenvectors;
              first vector starts in evecs[0],
              second vector starts in evecs[primme.n],
              third vector starts in evecs[primme.n*2]...  */
    primme_params primme;

    mpi_init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    read_infile(argc, argv);

    check_machine();
    geometry();
    check_files();
    print_info();
    start_ranlux(0, 1234);

    wsize(&nws, &nwv, &nwvd);
    alloc_ws(nws);
    alloc_wsd(nws + 8 + evadat.nev);
    alloc_wv(nwv);
    alloc_wvd(nwvd);
    status = alloc_std_status();
    w2 = malloc(evadat.nev * sizeof(double));
    error(w2 == NULL, 1, "eva_primme_dphi.c", "Error: Unable to allocate the primme w2 (normalization weights) \n");

#if (defined _OPENMP)
    error(omp_get_num_threads() != 1, 1, "eva_primme_dphi.c",
          "At the present stage eva_primme works only with OMP_NUM_THREADS=1");
#endif
    /*Set default values in PRIMME configuration struct */

    primme_initialize(&primme);

    /* Set problem matrix */
    if (evadat.opid == EVA_DW) {
        primme.matrixMatvec = MatMult_Dw_primme;
        primme.applyPreconditioner = MatMult_Dw_primme_Preconditioner;
        nsites = VOLUME;
    } else if (evadat.opid == EVA_QHAT) {
        primme.matrixMatvec = MatMult_Dw_primme;
        primme.applyPreconditioner = MatMult_Dw_primme_Preconditioner;
        nsites = VOLUME;
    }

    /*Function that implements the matrix-vector product
     A*x for solving the problem A*x = l*x */
    primme.n = 12 * nsites * NPROC; /* set problem dimension */
    primme.numEvals = evadat.nev; /* Number of wanted eigenpairs */
    primme.eps = evadat.tol; /* ||r|| <= eps * ||matrix|| */
    primme.target = evadat.target;

    primme.numTargetShifts = 1;
    primme.targetShifts = (double *)malloc(primme.numTargetShifts * sizeof(double));
    error(primme.targetShifts == NULL, 1, "eva_primme_dphi.c", "Error: Unable to allocate the primme.targetShifts \n");
    primme.targetShifts[0] = 0.0;
    primme.initSize = 0;
    /* primme.initSize may be not zero after a d/zprimme;
        so set it to zero to avoid the already converged eigenvectors
        being used as initial vectors. */

    /* DYNAMIC uses a runtime heuristic to choose the fastest method between
   PRIMME_DEFAULT_MIN_TIME and PRIMME_DEFAULT_MIN_MATVECS. But you can
   set another method, such as PRIMME_LOBPCG_OrthoBasis_Window, directly */
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_size(MPI_COMM_WORLD, &primme.numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &primme.procID);
    primme.commInfo = &comm; /* User-defined member to pass the communicator to
            globalSumReal and broadcastReal */
    /* In this example, the matrix is distributed by rows, and the first
     * processes may have an extra row in order to distribute the reing rows
     * n % numProcs */
    primme.nLocal = primme.n / primme.numProcs; /* Number of local rows */
    primme.globalSumReal = par_GlobalSum;
    primme.broadcastReal = broadcastForDouble;
    /* Set method to solve the problem */
    primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);

    /* Display PRIMME configuration struct (optional) */
    if (my_rank == 0 && !append) { primme_display_params(primme); }

    /* Allocate space for converged Ritz values and residual norms */
    evals = (double *)malloc(primme.numEvals * sizeof(double));
    error(evals == NULL, 1, "eva_primme_dphi.c", "Error: Unable to allocate the primme eigenvalues \n");

    evecs = (PRIMME_COMPLEX_DOUBLE *)malloc(primme.nLocal * primme.numEvals * sizeof(PRIMME_COMPLEX_DOUBLE));
    error(evecs == NULL, 1, "eva_primme_dphi.c", "Error: Unable to allocate the primme eigenvectors \n");

    rnorms = (double *)malloc(primme.numEvals * sizeof(double));
    error(rnorms == NULL, 1, "eva_primme_dphi.c", "Error: Unable to allocate the primme rnorms \n");

    wscheck = reserve_wsd(2 + primme.numEvals);
    message("Reserving % d Ev\n", 2 + primme.numEvals);

    iend = 0;
    wtavg = 0.0;

    for (nc = first; (iend == 0) && (nc <= last); nc += step) {
        primme.initSize = 0;

        message("Configuration no %d\n", nc);
	
        sprintf(cnfg_file, "%sn%d", nbase, nc);
        read_flds(iodat, cnfg_file, 0x0, 0x1);
        set_ud_phase();
        lat_parms();
        m0 = lat_parms().m0[0];
        maxm0 = lat_parms().m0[1];

        message("Evaluating mass range from %lf to %lf (with the max condition enabled)\n", m0, lat_parms().m0[1]);
        lastrun = 0;
	double lasteval = -1.;
	
        while (lastrun < 2) {
            set_sw_parms(m0);
            message("Evaluating mass %lf\n", m0);

            is = query_flags(UD_PHASE_SET);
            if (is == 0) { set_ud_phase(); }

            dfl_modes2(ifail, status);

            if ((ifail[0] < -2) || (ifail[1] < 0)) {
                print_status("dfl_modes2", ifail, status);
                error_root(1, 1, "smd_reset_dfl [smd.c]", "Deflation subspace generation failed");
            }

            if (is == 0) { unset_ud_phase(); }

            sw_term(NO_PTS);

            MPI_Barrier(MPI_COMM_WORLD);

            message("Evaluating configuration no %d with m=%lf\n", nc, m0);

            wt1 = MPI_Wtime();

            /* Call primme  */
            primme.initSize = 0;

            ret = zprimme(evals, evecs, rnorms, &primme);
            wt2 = MPI_Wtime();

            error(ret != 0, 1, "eva_primme_dphi.c", "Error: primme returned with nonzero exit status: %d \n", ret);

            for (i = 0; i < primme.initSize; i++) {
                memcpy((void *)wscheck[0], (void *)(evecs + i * primme.nLocal), sizeof(PRIMME_COMPLEX_DOUBLE) * primme.nLocal);

                Dw_dble(0.0, wscheck[0], wscheck[1]);
                mulg5_dble(nsites, 0, wscheck[1]);
                mulr_spinor_add_dble(nsites, 0, wscheck[1], wscheck[0], -evals[i]);
                rqsm = norm_square_dble(nsites, 1, wscheck[1]);
                del = sqrt(rqsm.q[0]);

                dlambda = spinor_prod5_dble(nsites, 1, wscheck[0], wscheck[0]);
                message("Eval[%d]: %-22.15E rnorm: %-22.15E oQCD check: %-22.15E dlambda.re: %-22.15E \n", i + 1, evals[i],
                        rnorms[i], del, dlambda.re.q[0]);
            }

            message(" %d eigenpairs converged\n", primme.initSize);
            message("Tolerance : %-22.15E\n", primme.aNorm * primme.eps);
            message("Iterations: %-" PRIMME_INT_P "\n", primme.stats.numOuterIterations);
            message("Restarts  : %-" PRIMME_INT_P "\n", primme.stats.numRestarts);
            message("Matvecs   : %-" PRIMME_INT_P "\n", primme.stats.numMatvecs);
            message("Preconds  : %-" PRIMME_INT_P "\n", primme.stats.numPreconds);
            message("Orthogonalization Time : %g\n", primme.stats.timeOrtho);
            message("Matvec Time            : %g\n", primme.stats.timeMatvec);
            message("GlobalSum Time         : %g\n", primme.stats.timeGlobalSum);
            message("Broadcast Time         : %g\n", primme.stats.timeBroadcast);
            message("Total Time             : %g\n", primme.stats.elapsedTime);
            if (primme.stats.lockingIssue) {
                message("\nA locking problem has occurred.\n");
                message("Some eigenpairs do not have a residual norm less than the tolerance.\n");
                message("However, the subspace of evecs is accurate to the required tolerance.\n");
            }
            message("Configuration no %d m0=%lf fully processed in %.2e sec ", nc, m0, wt2 - wt1);
            message("(average = %.2e sec)\n\n", wtavg / (double)((nc) / step + 1));

            if (m0 != lat_parms().m0[0]) {
                message("Projection matrix of Eigenvects\n");

                for (i = 0; i < primme.initSize; i++) {
                    memcpy((void *)wscheck[0], (void *)(evecs + i * primme.nLocal),
                           sizeof(PRIMME_COMPLEX_DOUBLE) * primme.nLocal);
                    mulg5_dble(nsites, 0, wscheck[0]);
                    qr = norm_square_dble(nsites, 1, wscheck[0]);
                    w1 = sqrt(qr.q[0]);

                    message("|");
                    for (j = 0; j < primme.initSize; j++) {
                        qr = spinor_prod_re_dble(nsites, 1, wscheck[0], wscheck[j + 2]);
                        message(" %.2e ", (qr.q[0] / w1) / w2[j]);
                    }
                    message("|\n");
                }
            }

            for (i = 0; i < primme.initSize; i++) {
                memcpy((void *)wscheck[2 + i], (void *)(evecs + i * primme.nLocal),
                       sizeof(PRIMME_COMPLEX_DOUBLE) * primme.nLocal);
                mulg5_dble(nsites, 0, wscheck[i + 2]);
                qr = norm_square_dble(nsites, 1, wscheck[i + 2]);
                w2[i] = sqrt(qr.q[0]);
            }

            switch (primme.dynamicMethodSwitch) {
            case -1:
                message("Recommended method for next run: DEFAULT_MIN_MATVECS\n");
                break;
            case -2:
                message("Recommended method for next run: DEFAULT_MIN_TIME\n");
                break;
            case -3:
                message("Recommended method for next run: DYNAMIC (close call)\n");
                break;
            }

            /*determine the new m0 or stop:*/

            if (lastrun == 1) {
                message("End of iteration for configuration no %d\n", nc);

                break;
            }

	    if (lasteval == -1.)
	      lasteval = evals[0];
		    
            double ratio = evals[0] / lasteval;
            lasteval = evals[0];

            double mineval = ABS(evals[0]);
            for (i = 1; i < primme.initSize; i++) {
                if (mineval > ABS(evals[i])) { mineval = ABS(evals[i]); }
            }

            if (m0 == lat_parms().m0[0]) {
                starteval = mineval / 20;
                if (lat_parms().m0[0] + 20 * ABS(evals[1]) < lat_parms().m0[1]) {
                    maxm0 = lat_parms().m0[0] + 20 * ABS(evals[1]);
                    MPI_Bcast(&maxm0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    message("Updated the mass range to [%lf : %lf] (ude to the max condition m_end -m_start <= 20*min_ev)\n",
                            m0, maxm0);
                }
            }

            mineval = (starteval > mineval) ? starteval : mineval;

            if (m0 + mineval > maxm0) {
                m0 = maxm0;
                lastrun++;
            } else {
                m0 = (ratio > 0.) ? m0 + mineval : m0 + 2 * mineval;
            }
            message("Configuration no %d next step will evaluate m=%lf\n", nc, m0);
            MPI_Bcast(&m0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        release_wsd();

        check_endflag(&iend);

        if (my_rank == 0) {
            fflush(flog);
            copy_file(log_file, log_save);
            fclose(flog);
        }
    }

    MPI_Finalize();
    exit(0);
}

void MatMult_Dw_primme(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
    spinor_dble **wsd3;

    sw_term(NO_PTS);

    wsd3 = reserve_wsd(2);

    error(*blockSize != 1, 1, "MatMult_Dw_primme", "Blocksize must be 1 instead found (%d)\n", *blockSize);

    memcpy((void *)wsd3[0], (void *)x, sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);

    Dw_dble(0.0, wsd3[0], wsd3[1]);

    mulg5_dble(VOLUME, 2, (spinor_dble *)wsd3[1]);
    memcpy((void *)y, (void *)wsd3[1], sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);

    release_wsd();
}

static void par_GlobalSum(void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr) {
    MPI_Comm communicator = *(MPI_Comm *)primme->commInfo;

    if (sendBuf == recvBuf) {
        *ierr = MPI_Allreduce(MPI_IN_PLACE, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    } else {
        *ierr = MPI_Allreduce(sendBuf, recvBuf, *count, MPI_DOUBLE, MPI_SUM, communicator) != MPI_SUCCESS;
    }
}

static void broadcastForDouble(void *buffer, int *count, primme_params *primme, int *ierr) {
    MPI_Comm communicator = *(MPI_Comm *)primme->commInfo;

    if (MPI_Bcast(buffer, *count, MPI_DOUBLE, 0 /* root */, communicator) == MPI_SUCCESS) {
        *ierr = 0;
    } else {
        *ierr = 1;
    }
}

void MatMult_Dw_primme_Preconditioner(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme,
                                      int *ierr) {
    spinor_dble **wsd3;
    static int *status = NULL;
    int ifail[2];
    sap_parms_t sap;

    lat_parms_t lat = lat_parms();
    dfl_pro_parms_t dfp = dfl_pro_parms();
    dfl_gen_parms_t dfg = dfl_gen_parms();

    if (status == NULL) { status = alloc_std_status(); }

    sap = sap_parms();
    set_sap_parms(sap.bs, 1, 4, 5);
    wsd3 = reserve_wsd(2);

    memcpy((void *)wsd3[0], (void *)x, sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);
    mulg5_dble(VOLUME, 2, wsd3[0]);

    dfl_sap_gcr2(dfp.nkv, dfp.nmx, lat.isw, dfp.res, dfg.mu, wsd3[0], wsd3[1], ifail, status);

    memcpy((void *)y, (void *)wsd3[1], sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);

    release_wsd();
}
