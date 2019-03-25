/*****************************************************************************\
  SWESol.c:
  Solve Shallow Water Equations using GGDML language Extensions
  Nabeeh Jum'ah & Prof. John Thuburn
\*****************************************************************************/

#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Fields defined using GGDML specifiers
float CELL 2D f_H;      /* Surface level and tendency  */
float CELL 2D f_HT;
float CELL 2D f_B;
float EDGE 2D f_U;      /* Velocity and tendencies */
float EDGE 2D f_UT;
float EDGE 2D f_V;
float EDGE 2D f_VT;
float EDGE 2D f_F;      /* Flux */
float EDGE 2D f_G;

// Scalars
const float dx = 1.0;   /* Space and time intervals */
const float dy = 1.0;
const float dt = 0.001;
const float g  = 9.8;   /* Gravity */
const float f  = 0.1;   /* Coriolis */

/*****************************************************************************\
  Compute flux:
  Compute both X ynd Y components of the flux
\*****************************************************************************/

void compute_flux()
{
    // Compute the flux component in the X dimension
    foreach e IN grid {

        // Use GGDML access operators east_cell & west_cell
        //  to refer to the cells sharing the edge
        f_F[e] = f_U[e] * (f_H[e.east_cell()] +
                           f_H[e.west_cell()]) / 2.0;
    }

    // compute the flux component in the Y dimension
    foreach e in grid {

        // Use GGDML access operators north_cell & south_cell
        // to refer to the cells sharing the edge
        f_G[e] = f_V[e] * (f_H[e.north_cell()] +
                           f_H[e.south_cell()]) / 2.0;
    }
}

/*****************************************************************************\
  compute_U_tendency:
  Compute tendency of the velocity in the X dimension
\*****************************************************************************/

void compute_U_tendency()
{
    // Compute different terms of tendency
    // Use GGDML iterator to traverse the edges where the U-Tendency is located
    foreach e in grid {

        // Use GGDML access operators edge_????_neighbor
        // to refer to the neiboring U edges in X direction
        float udux = f_U[e] * (f_U[e.edge_east_neighbor()] -
                               f_U[e.edge_west_neighbor()]) / (2.0 * dx);

        // Use GGDML access operators edge_??_neighbor
        // to refer to the neiboring V edges
        float vbar = (f_V[e.edge_ne_neighbor()] +
                      f_V[e.edge_nw_neighbor()] +
                      f_V[e.edge_se_neighbor()] +
                      f_V[e.edge_sw_neighbor()]) / 4.0;

        // Use GGDML access operators edge_v?????_neighbor
        // to refer to the neiboring U edges in Y direction
        float vduy = vbar * (f_U[e.edge_vnorth_neighbor()] -
                             f_U[e.edge_vsouth_neighbor()]) / (2.0 * dy);

        // Use GGDML access operators east_cell & west_cell
        //  to refer to the cells sharing the edge
        float gdhbx = g * (f_H[e.east_cell()] +
                           f_B[e.east_cell()] -
                           f_H[e.west_cell()] -
                           f_B[e.west_cell()]) / dx;

        float fvbar = f * vbar;
        f_UT[e] = fvbar - udux - vduy - gdhbx;
    }
}

/*****************************************************************************\
  update_U:
  Update the velocity in the X dimension
\*****************************************************************************/

void update_U()
{
    foreach e in grid {
        f_U[e] = f_U[e] + f_UT[e] * dt;
    }
}

/*****************************************************************************\
  compute_V_tendency:
  Compute tendency of the velocity in the Y dimension
\*****************************************************************************/

void compute_V_tendency()
{
    // Compute different terms of tendency
    // Use GGDML iterator to traverse the edges where the V-Tendency is located
    foreach e in grid {
        // Use GGDML access operators edge_?????_neighbor
        // to refer to the neiboring V edges in Y direction
        float vdvy = f_V[e] * (f_V[e.edge_north_neighbor()] -
                     f_V[e.edge_south_neighbor()]) / (2.0 * dy);

        // Use GGDML access operators edge_??_neighbor
        // to refer to the neiboring U edges
        float ubar = (f_U[e.edge_en_neighbor()] +
                      f_U[e.edge_es_neighbor()] +
                      f_U[e.edge_wn_neighbor()] +
                      f_U[e.edge_ws_neighbor()]) / 4.0;

        // Use GGDML access operators edge_h????_neighbor
        // to refer to the neiboring V edges in X direction
        float udvx = ubar * (f_V[e.edge_heast_neighbor()] -
                             f_V[e.edge_hwest_neighbor()]) / (2.0 * dx);

        // Use GGDML access operators north_cell & south_cell
        //  to refer to the cells sharing the edge
        float gdhby = g * (f_H[e.north_cell()] +
                           f_B[e.north_cell()] -
                           f_H[e.south_cell()] -
                           f_B[e.south_cell()]) / dy;

        float fubar = f * ubar;
        f_VT[e] = 0.0 - vdvy - udvx - gdhby - fubar;
    }
}

/*****************************************************************************\
  update_V:
  Update the velocity in the Y dimension
\*****************************************************************************/

void update_V()
{
    foreach e in grid {
        f_V[e] = f_V[e] + f_VT[e] * dt;
    }
}

/*****************************************************************************\
  compute_H_tendency:
  Compute tendency of the surface level
\*****************************************************************************/

void compute_H_tendency()
{
    // Compute the two terms of tendency
    // Use GGDML iterator to traverse the grid cells
    foreach c in grid {

        // Use GGDML access operators east_edge & west_edge
        //  to refer to the U edges of the cell
        float df = (f_F[c.east_edge()] -
                    f_F[c.west_edge()]) / dx;

        // Use GGDML access operators north_edge & south_edge
        //  to refer to the V edges of the cell
        float dg = (f_G[c.north_edge()] -
                    f_G[c.south_edge()]) / dy;

        f_HT[c] = df + dg;
    }
}

/*****************************************************************************\
  update_H:
  Update the surface level
\*****************************************************************************/

void update_H()
{
    // Update the surface level
    // Use GGDML iterator to traverse the grid cells
    foreach c in grid {
        f_H[c] = f_H[c] - dt * f_HT[c];
    }
}

/*****************************************************************************\
  update_values:
  Call the tendecies computations and the update kernels
\*****************************************************************************/

void update_values()
{
    compute_U_tendency();
    update_U();
    compute_V_tendency();
    update_V();
    compute_H_tendency();
    update_H();
}

/*****************************************************************************\
  time_sec:
  A helper function to measure time
  It returns a floating point value
  Differnce between two calls allows measuring the running time of the code
\*****************************************************************************/

double time_sec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
}

/*****************************************************************************\
  main:
  The main entry point for the code
  It allocates and deallocates the memory for the fields, and runs the
  time-step loop
\*****************************************************************************/

#define TIMESTEPS 1000

int main(int argc, char **argv)
{
    // Initialize necessary libraries
    INITCOMMLIB;
    INITCOMM;

    // Allocate necessary memory for the fields
    ALLOC f_H;
    ALLOC f_HT;
    ALLOC f_U;
    ALLOC f_UT;
    ALLOC f_V;
    ALLOC f_VT;
    ALLOC f_B;
    ALLOC f_F;
    ALLOC f_G;

    int time_step = 0;
    double run_time = time_sec();

    //time stepping loop
    timestep(time_step = 0;
             time_step < TIMESTEPS;
             time_step++) {

        // Compute flux
        compute_flux();

        // Compute tendencies and update values
        update_values();
    }
    run_time = time_sec() - run_time;
    printf("%f,%f\n",
           run_time,
           60.0 * GRIDX * GRIDY * TIMESTEPS / run_time / 1000000000);

    // Deallocate memory
    DEALLOC f_H;
    DEALLOC f_HT;
    DEALLOC f_U;
    DEALLOC f_UT;
    DEALLOC f_V;
    DEALLOC f_VT;
    DEALLOC f_B;
    DEALLOC f_F;
    DEALLOC f_G;

    // Finalize necessary libraries
    FINCOMMLIB;

    return 0;
}
