
/*******************************************************************************
*
* File eva_primme_new.c
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*
* Syntax: eva_primme_new -i <input file>
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

#include "primme.h" /* header file is required to run primme */

#define N0 (NPROC0 * L0)
#define N1 (NPROC1 * L1)
#define N2 (NPROC2 * L2)
#define N3 (NPROC3 * L3)

void MatMult_Qhat_primme(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr);
void MatMult_Qhat_primme_Preconditioner(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
                                        primme_params *primme, int *ierr);

static void par_GlobalSum(void *sendBuf, void *recvBuf, int *count, primme_params *primme, int *ierr);
static void broadcastForDouble(void *buffer, int *count, primme_params *primme, int *ierr);
int mpierr;

typedef enum { EVA_QHAT } evaop_t;

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

static int my_rank, endian;
static int first, last, step;
static int ifail0[2];

static iodat_t iodat[1];
static char nbase[NAME_SIZE], log_dir[NAME_SIZE];
static char log_file[NAME_SIZE], log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE], end_file[NAME_SIZE];
static FILE *fin = NULL, *flog = NULL, *fend = NULL;

static void read_dirs(void) {
    if (my_rank == 0) {
        find_section("Run name");
        read_line("name", "%s", nbase);

        find_section("Log directory");
        read_line("log_dir", "%s", log_dir);
    }

    MPI_Bcast(nbase, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(log_dir, NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
}

static void setup_files(void) {
    error(name_size("%s/%s.eva_primme.log~", log_dir, nbase) >= NAME_SIZE, 1, "setup_files [eva_primme.c]",
          "log_dir name is too long");

    sprintf(log_file, "%s/%s.eva_primme.log", log_dir, nbase);
    sprintf(end_file, "%s/%s.eva_primme.end", log_dir, nbase);
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
                   "read_cnfg_range [eva_primme.c]", "Improper configuration range");
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

        error_root((nev < 1) || (tol < 0.0), 1, "read_primme_parms [eva_spec.c]", "Parameters are out of range");

        read_line("operator", "%s", name);
        read_line("target", "%s", targname);

        if (strcmp(name, "Qhat") == 0) {
            opid = EVA_QHAT;
        } else {
            error_root(1, 1, "read_primme_parms [eva_spec.c]", "Unknown matrix type");
        }

        if (strcmp(targname, "small") == 0) {
            targ = primme_closest_abs;
        } else if (strcmp(targname, "large") == 0) {
            targ = primme_largest_abs;
        } else {
            error_root(1, 1, "read_primme_parms [eva_spec.c]", "Unknown eigenvalue search target");
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

        error_root((ifile == 0) || (ifile == (argc - 1)), 1, "read_infile [eva_primme.c]",
                   "Syntax: eva_primme -i <input file>");

        error_root(endian == UNKNOWN_ENDIAN, 1, "read_infile [eva_primme.c]", "Machine has unknown endianness");

        fin = freopen(argv[ifile + 1], "r", stdin);
        error_root(fin == NULL, 1, "read_infile [eva_primme.c]", "Unable to open input file");
    }

    MPI_Bcast(&endian, 1, MPI_INT, 0, MPI_COMM_WORLD);

    read_dirs();
    read_iodat("Configurations", "i", iodat);
    read_cnfg_range();
    read_lat_parms("Dirac operator", 0x2);
    read_bc_parms("Boundary conditions", 0x2);

    read_sap_parms("SAP", 0x1);

    read_dfl_parms("Deflation subspace");
    read_dfl_pro_parms("Deflation projection");
    read_dfl_gen_parms("Deflation subspace generation");

    setup_files();

    if (my_rank == 0) { fclose(fin); }
}

static void check_files(void) {
    int ie;

    if (my_rank == 0) {
        ie = check_file(log_file, "r");
        error_root(ie != 0, 1, "check_files [eva_primme.c]", "Attempt to overwrite old *.log file");
    }

    error(name_size("%sn%d", nbase, last) >= NAME_SIZE, 1, "check_files [eva_primme.c]", "Configuration base name is too long");
    sprintf(cnfg_file, "%sn%d", nbase, last);
    check_iodat(iodat, "i", 0x1, cnfg_file);
}

static void print_info(void) {
    int isap, idfl;
    long ip;

    if (my_rank == 0) {
        ip = ftell(flog);
        fclose(flog);

        if (ip == 0L) { remove("STARTUP_ERROR"); }

        flog = freopen(log_file, "w", stdout);
        error_root(flog == NULL, 1, "print_info [eva_primme.c]", "Unable to open log file");
        printf("\n");

        printf("Eigenvalue spectrum calculation of selected Dirac operator with Primme\n");
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

        if (isap) { print_sap_parms(0x0); }

        if (idfl) { print_dfl_parms(0x0); }

        print_iodat("i", iodat);
        printf("Configurations no %d -> %d in steps of %d\n\n", first, last, step);
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

    maxn(nws, dp.Ns + 2);
    maxn(nwv, 2 * dpr.nkv + 3);
    maxn(nwvd, 4);
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
    int nc, iend, *status, ret, i;
    int nws, nwv, nwvd;

    double wt1, wt2, wtavg;
    dfl_parms_t dfl;
    spinor_dble **wscheck;
    complex_qflt dlambda;
    pauli_dble *m;
    qflt rqsm;

    double del;

    pauli_wsp_t *pwsp;

    /* PRIMME configuration struct */

    double *evals; /* Array with the computed eigenvalues */
    double *rnorms; /* Array with the computed eigenpairs residual norms */
    PRIMME_COMPLEX_DOUBLE *evecs; /* Array with the computed eigenvectors;
              first vector starts in evecs[0],
              second vector starts in evecs[primme.n],
              third vector starts in evecs[primme.n*2]...  */
    primme_params primme;

    mpi_init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    read_infile(argc, argv);
    read_primme_parms();
    check_machine();
    geometry();
    check_files();
    print_info();
    start_ranlux(0, 1234);

    wsize(&nws, &nwv, &nwvd);
    alloc_ws(nws);
    wsd_uses_ws();
    alloc_wv(nwv);
    alloc_wvd(nwvd);
    status = alloc_std_status();
    pwsp = alloc_pauli_wsp();

    dfl = dfl_parms();
    /*Set default values in PRIMME configuration struct */

    primme_initialize(&primme);

    /* Set problem matrix */
    primme.matrixMatvec = MatMult_Qhat_primme;
    primme.applyPreconditioner = MatMult_Qhat_primme_Preconditioner;
    /*Function that implements the matrix-vector product
     A*x for solving the problem A*x = l*x */

    primme.n = 12 * N0 * N1 * N2 * N3 / 2; /* set problem dimension */
    primme.numEvals = evadat.nev; /* Number of wanted eigenpairs */
    primme.eps = evadat.tol; /* ||r|| <= eps * ||matrix|| */
    primme.target = evadat.target;

    primme.numTargetShifts = 1;
    primme.targetShifts = (double *)malloc(primme.numTargetShifts * sizeof(double));
    primme.targetShifts[0] = 0.0;
    primme.initSize = 0;
    /* primme.initSize may be not zero after a d/zprimme;
        so set it to zero to avoid the already converged eigenvectors
        being used as initial vectors. */

    /* Set method to solve the problem */
    primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);
    /* DYNAMIC uses a runtime heuristic to choose the fastest method between
     PRIMME_DEFAULT_MIN_TIME and PRIMME_DEFAULT_MIN_MATVECS. But you can
     set another method, such as PRIMME_LOBPCG_OrthoBasis_Window, directly */
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_size(MPI_COMM_WORLD, &primme.numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &primme.procID);
    primme.commInfo = &comm; /* User-defined member to pass the communicator to
            globalSumReal and broadcastReal */
    /* In this example, the matrix is distributed by rows, and the first
   * processes may have an extra row in order to distribute the remaining rows
   * n % numProcs */
    PRIMME_INT nLocal = primme.n / primme.numProcs + (primme.n % primme.numProcs > primme.procID ? 1 : 0);
    primme.nLocal = nLocal; /* Number of local rows */
    primme.globalSumReal = par_GlobalSum;
    primme.broadcastReal = broadcastForDouble;

    /* Display PRIMME configuration struct (optional) */
    if (my_rank == 0) { primme_display_params(primme); }

    /* Allocate space for converged Ritz values and residual norms */
    evals = (double *)malloc(primme.numEvals * sizeof(double));
    evecs = (PRIMME_COMPLEX_DOUBLE *)malloc(primme.nLocal * primme.numEvals * sizeof(PRIMME_COMPLEX_DOUBLE));
    rnorms = (double *)malloc(primme.numEvals * sizeof(double));

    wscheck = reserve_wsd(4);
    iend = 0;
    wtavg = 0.0;

    for (nc = first; (iend == 0) && (nc <= last); nc += step) {
        primme.initSize = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        wt1 = MPI_Wtime();

        if (my_rank == 0) {
            printf("Configuration no %d\n", nc);
            fflush(flog);
        }

        sprintf(cnfg_file, "%sn%d", nbase, nc);
        read_flds(iodat, cnfg_file, 0x0, 0x1);
        set_ud_phase();

        if (dfl.Ns) {
            dfl_modes2(ifail0, status);

            if ((ifail0[0] < -2) || (ifail0[1] < 0)) {
                print_status("dfl_modes2", ifail0, status);
                error_root(1, 1, "main [eva_primme.c]", "Deflation subspace generation failed");
            }
        }

        /* Call primme  */
        ret = zprimme(evals, evecs, rnorms, &primme);

        if (my_rank == 0) {
            if (ret != 0) {
                printf("Error: primme returned with nonzero exit status: %d \n", ret);
                return -1;
            }
        }

        set_sd2zero(VOLUME_TRD / 2, 2, wscheck[0] + VOLUME / 2);
        set_sd2zero(VOLUME_TRD / 2, 2, wscheck[1] + VOLUME / 2);
        set_sd2zero(VOLUME_TRD / 2, 2, wscheck[2] + VOLUME / 2);
        set_sd2zero(VOLUME_TRD / 2, 2, wscheck[3] + VOLUME / 2);

        for (i = 0; i < primme.initSize; i++) {
            memcpy((void *)wscheck[0], (void *)(evecs + i * primme.nLocal), sizeof(PRIMME_COMPLEX_DOUBLE) * primme.nLocal);

            Dwhat_dble(0.0, wscheck[0], wscheck[1]);
            mulg5_dble(VOLUME_TRD / 2, 2, wscheck[1]);
            mulr_spinor_add_dble(VOLUME_TRD / 2, 2, wscheck[1], wscheck[0], -evals[i]);
            rqsm = norm_square_dble(VOLUME / 2, i, wscheck[1]);
            del = sqrt(rqsm.q[0]);

            m = swdfld();
            sw_term(ODD_PTS);
            Dwoe_dble(wscheck[0], wscheck[0]);
            apply_swinv_dble(VOLUME / 2, 0.0, m, wscheck[0], pwsp, wscheck[0]);

            dlambda = spinor_prod5_dble(VOLUME / 2, i, wscheck[0], wscheck[0]);

            if (my_rank == 0) {
                printf("Eval[%d]: %-22.15E rnorm: %-22.15E oQCD check: %-22.15E dlambda.re: %-22.15E \n", i + 1, evals[i],
                       rnorms[i], del, dlambda.re.q[0]);
            }
        }

        if (my_rank == 0) {
            printf(" %d eigenpairs converged\n", primme.initSize);
            printf("Tolerance : %-22.15E\n", primme.aNorm * primme.eps);
            printf("Iterations: %-" PRIMME_INT_P "\n", primme.stats.numOuterIterations);
            printf("Restarts  : %-" PRIMME_INT_P "\n", primme.stats.numRestarts);
            printf("Matvecs   : %-" PRIMME_INT_P "\n", primme.stats.numMatvecs);
            printf("Preconds  : %-" PRIMME_INT_P "\n", primme.stats.numPreconds);
            printf("Orthogonalization Time : %g\n", primme.stats.timeOrtho);
            printf("Matvec Time            : %g\n", primme.stats.timeMatvec);
            printf("GlobalSum Time         : %g\n", primme.stats.timeGlobalSum);
            printf("Broadcast Time         : %g\n", primme.stats.timeBroadcast);
            printf("Total Time             : %g\n", primme.stats.elapsedTime);
            if (primme.stats.lockingIssue) {
                printf("\nA locking problem has occurred.\n");
                printf("Some eigenpairs do not have a residual norm less than the tolerance.\n");
                printf("However, the subspace of evecs is accurate to the required tolerance.\n");
            }

            switch (primme.dynamicMethodSwitch) {
            case -1:
                printf("Recommended method for next run: DEFAULT_MIN_MATVECS\n");
                break;
            case -2:
                printf("Recommended method for next run: DEFAULT_MIN_TIME\n");
                break;
            case -3:
                printf("Recommended method for next run: DYNAMIC (close call)\n");
                break;
            }
        }

        if (my_rank == 0) {
            printf("Configuration no %d fully processed in %.2e sec ", nc, wt2 - wt1);
            printf("(average = %.2e sec)\n\n", wtavg / (double)((nc) / step + 1));
        }

        check_endflag(&iend);
    }

    if (my_rank == 0) {
        fflush(flog);
        copy_file(log_file, log_save);
        fclose(flog);
    }

    MPI_Finalize();
    exit(0);
}

void MatMult_Qhat_primme(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize, primme_params *primme, int *ierr) {
    spinor_dble **wsd3;

    int blk; /* nx */
    PRIMME_COMPLEX_DOUBLE *xvec; /* pointer to i-th input vector x */
    PRIMME_COMPLEX_DOUBLE *yvec; /* pointer to i-th output vector y */
    sw_term(ODD_PTS);

    wsd3 = reserve_wsd(2);
    for (blk = 0; blk < *blockSize; blk++) {
        xvec = (PRIMME_COMPLEX_DOUBLE *)x + *ldx * blk; /* pointer to i-th input vector x */
        yvec = (PRIMME_COMPLEX_DOUBLE *)y + *ldy * blk; /* pointer to i-th output vector y */

        memcpy((void *)wsd3[0], (void *)xvec, sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);

        Dwhat_dble(0.0, wsd3[0], wsd3[1]);
        mulg5_dble(VOLUME_TRD / 2, 2, wsd3[1]);

        memcpy((void *)yvec, (void *)wsd3[1], sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);
    }
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

void MatMult_Qhat_primme_Preconditioner(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
                                        primme_params *primme, int *ierr) {
    spinor_dble **wsd3;
    int status;
    int ifail;

    int blk; /* nx */
    PRIMME_COMPLEX_DOUBLE *xvec; /* pointer to i-th input vector x */
    PRIMME_COMPLEX_DOUBLE *yvec; /* pointer to i-th output vector y */
    lat_parms_t lat = lat_parms();
    dfl_pro_parms_t dfp = dfl_pro_parms();
    dfl_gen_parms_t dfg = dfl_gen_parms();

    wsd3 = reserve_wsd(1);
    for (blk = 0; blk < *blockSize; blk++) {
        xvec = (PRIMME_COMPLEX_DOUBLE *)x + *ldx * blk; /* pointer to i-th input vector x */
        yvec = (PRIMME_COMPLEX_DOUBLE *)y + *ldy * blk; /* pointer to i-th output vector y */

        set_sd2zero(VOLUME_TRD / 2, 2, wsd3[0] + VOLUME / 2);
        memcpy((void *)wsd3[0], (void *)xvec, sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);

        mulg5_dble(VOLUME_TRD / 2, 2, wsd3[0]);

        dfl_sap_gcr(dfp.nkv, dfp.nmx, lat.isw, dfp.res, dfg.mu, wsd3[0], wsd3[0], &ifail, &status);

        memcpy((void *)yvec, (void *)wsd3[0], sizeof(PRIMME_COMPLEX_DOUBLE) * *ldx);
    }

    release_wsd();
}
