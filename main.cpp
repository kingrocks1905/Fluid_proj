#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <sstream>

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "lbfgs.h"
#include "cmath"
#include <random>

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0.0, 1.0);

double sqr(double x) { return x * x; };

class Vector {
public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
    }
    double norm2() const {
        return data[0] * data[0] + data[1] * data[1];
    }
    double norm() const {
        return sqrt(norm2());
    }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double& operator[](int i) { return data[i]; };
    double data[2];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector& b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector& a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector& a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1];
}


class Polygon {
public:

    double area() {
        if (vertices.size() < 3) return 0;
        // TODO Lab 2
        // Compute the area of the polygon

        double area = 0.0;

        for(size_t i = 0; i < vertices.size(); i++){

            const Vector& vi = vertices[i];

            const Vector& vj = vertices[(i+1)% vertices.size()];

            area += vi[0] * vj[1] - vj[0] * vi[1];
        }
        return std::abs(area * 0.5);
    }

    Vector centroid() {
        if (vertices.size() < 3) return Vector(0, 0);
        // TODO Lab 3
        // Compute the centroid of the polygon

        Vector c(0,0);

        double a = 0.0;

        for(size_t i = 0; i < vertices.size(); i++){

            const Vector& vi = vertices[i];

            const Vector& vj = vertices[(i + 1) % vertices.size()];

            double cross = vi[0] * vj[1] - vj[0] * vi[1];

            a = a+cross;

            c = c + (vi + vj) * cross;

        }

        return c / (3.0 * a);
    
    }

    double integral_square_distance(const Vector& Pi) {
        if (vertices.size() < 3) return 0;

        // TODO Lab 2
        // Compute the integral of ||x-Pi||^2 over the polygon

        double integral = 0;

        // We will try to tringulate the polygon using Pi as pivot

        for (size_t i = 0; i< vertices.size(); i++){

            const Vector& vi = vertices[i];

            const Vector& vj = vertices[(i+1)% vertices.size()];

            Vector a = vi - Pi;
            Vector b = vj- Pi;

            // now we'll calculate the area of thsi triangle

            double tri_area= std::abs(a[0]*b[1] - b[0]* a[1])* 0.5;

            integral += (tri_area / 6.0) * (a.norm2() + dot(a,b)+ b.norm2());

        

        }



        return integral;
    }

    std::vector<Vector> vertices;
};


void save_frame(const std::vector<Polygon>& cells, std::string filename, int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto& V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3) continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 && det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge) edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        }
        else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}


class VoronoiDiagram {

public:

    VoronoiDiagram() {
    };


    void compute() {

        // TODO Lab 1 (Voronoi)
        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (Lab 3, fluids) : also clip it by a disk of radius sqrt(w_i - w_air) centered at Pi


        cells.resize(points.size());

       
        
        double w_air = weights.back(); // We'll add the very last ray in the array to Air

        for (int i = 0; i < points.size(); ++i) {
            
            Polygon cell;

            // I made the pushbacks here instead of the main function


            cell.vertices.push_back(Vector(0, 0));
            cell.vertices.push_back(Vector(1, 0));
            cell.vertices.push_back(Vector(1, 1));
            cell.vertices.push_back(Vector(0, 1));

            // For every other site we will clip with the bisector
            for (int j = 0; j < points.size(); ++j) {

                if (i == j) continue; 
                
                cell = clip_by_bisector(cell, points[i], points[j], weights[i], weights[j]);
            }

            //Now we'll clip against the air to form something like a free surface

            if (weights[i]<= w_air){

                // If this condition happens then our radius is imaginary
                cell.vertices.clear();
            }

            else{
                double radius = sqrt(weights[i]- w_air);

                int number_sides = 32; // We'll have a 32 sided polygon to approximate

                for (int k = 0;k< number_sides; k++){

                    double theta1 = k* 2.0 *M_PI/ number_sides;
                    double theta2 = (k+1)* 2.0 *M_PI/ number_sides;

                    Vector u = points[i] + Vector(radius * cos(theta1), radius * sin(theta1));
                    Vector v = points[i] +Vector(radius *cos(theta2), radius * sin(theta2));

                    cell = clip_by_edge(cell,u,v);
                }

            }
            cells[i] = cell;
        }
    }


    static Polygon clip_by_edge(const Polygon& V, const Vector& u, const Vector& v) {

        // TODO Lab 3 (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a (discretized) disk

        Polygon result;

        if (V.vertices.empty()) return result;

        // Here the normal pointing to inside
        Vector N(v[1] - u[1], u[0] - v[0]); 


        for (size_t j = 0; j < V.vertices.size(); j++) {


            const Vector& A = V.vertices[j];

            const Vector& B = V.vertices[(j + 1) % V.vertices.size()];

            double dotA = dot(A - u, N);

            double dotB = dot(B - u, N);

            bool A_inside = (dotA <= 0);
            bool B_inside = (dotB <= 0);

            if (A_inside != B_inside) {
                double t = dotA / (dotA - dotB);
                Vector intersection = A + (B - A) * t;

                if (A_inside && !B_inside) {
                    result.vertices.push_back(intersection);
                } 
                
                else if (!A_inside && B_inside) {

                    result.vertices.push_back(intersection);
                    result.vertices.push_back(B);
                }

            } else if (A_inside && B_inside) {

                result.vertices.push_back(B);
            }
        }

    




        return result;
    }

    
    static Polygon clip_by_bisector(const Polygon& V, const Vector& P0, const Vector& Pi, double w0, double wi) {

        // TODO Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the current site of the Voronoi cell being computed) and Pi (another site)
        
        // TODO Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre cells, i.e., w0 != w1

        Polygon result;

        if (V.vertices.empty()) {
            return result;

        }

        // M here would be the midpoint between the two points
        Vector M = (P0 + Pi) * 0.5;

        // N would be the Normal pointing towards Pi 
        Vector N = Pi - P0; 

        double dist2 = N.norm2();

        // I want to prevent division by 0 over here

        if (dist2 < 1e-23131){
            return V;
        }

        
        // This would be the shifted midpoint for the power diagram
        Vector M_shifted  = M + (w0-wi)/ (2.0 * dist2) * N; 
        


        // Now we'll loop over all the edges in the polygon

        for (size_t j = 0; j < V.vertices.size(); j++) {

            const Vector& A = V.vertices[j];

            const Vector& B = V.vertices[(j + 1) % V.vertices.size()];

            double dotA = dot(A - M_shifted, N);
            double dotB = dot(B - M_shifted, N);

            bool A_inside = (dotA <= 0);
            bool B_inside = (dotB <= 0);

            // We now compute the exact intersection point if the edge crosses the boundary

            Vector intersection;

            if (A_inside != B_inside) {
               
                double t = dotA / (dotA - dotB); 

                intersection = A + (B - A) * t; 
            }

            // We apply the Sutherland-Hodgman logic for all the four cases

            if (A_inside && B_inside) {
                // If  A is inside and B is also inside then we add B

                result.vertices.push_back(B);

            } else if (A_inside && !B_inside) {
                // If A is  inside, B is outside then we  Add M (Intersection)

                result.vertices.push_back(intersection);

            } else if (!A_inside && B_inside) {

                result.vertices.push_back(intersection);
                result.vertices.push_back(B);
            } 
            else {

                // If A and B are both outside we do nothing


            }
        }


        return result;
    }


    std::vector<Vector> points;    // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights;   // Lab 2 (OT) : the weight associated to each site (the Laguerre weight, i.e. the dual optimal transport variables to be optimized)
    
    std::vector<Polygon> cells;   // Lab 1 : the polygons representing each individual cell

};


// Lab 2 
class OptimalTransport {

public:
    OptimalTransport() {};

    void optimize();

    VoronoiDiagram vor;

    double fluid_volume = 0.4;
};


// Labs 2 and 3
static lbfgsfloatval_t evaluate(
    void* instance,
    const lbfgsfloatval_t* x,
    lbfgsfloatval_t* g,
    const int n,
    const lbfgsfloatval_t step
)
{
    OptimalTransport* ot = (OptimalTransport*)(instance);

    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();
  
   
    // Lab 2 (Optimal transport) : compute the function to be minimized (fx) and its gradient (g[i], i=0..n-1)
    // Lab 3 (fluid) : adapt these functions to support partial optimal transport (now "n" has been increased by 1 to account for the air variable)
    
    lbfgsfloatval_t fx = 0.0;

    int N_particles = n-1; 

    double target_fluid_area = ot->fluid_volume / N_particles;

    double target_air_area = 1.0 - ot->fluid_volume;

    double total_fluid_area = 0.0;

    // Classic particle calculations to do here
    for (int i = 0; i < N_particles; i++) {

        double w_i = x[i];

        double area_i = ot->vor.cells[i].area();
        total_fluid_area += area_i;
        
        double integral_sq = ot->vor.cells[i].integral_square_distance(ot->vor.points[i]);
        
        g[i] = area_i - target_fluid_area;
        fx += (w_i * area_i) - (w_i * target_fluid_area) - integral_sq;
    }
    
    // Time for the air calculations 
    double w_air = x[N_particles];

    double area_air = 1.0 - total_fluid_area; 
    
    g[N_particles] = area_air - target_air_area;

    fx += (w_air * area_air) - (w_air * target_air_area); 

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(
    void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}


// Lab 2
void OptimalTransport::optimize() {

    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);

    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer
    int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress, (void*)this, &param);

    // copy the result back to the voronoi structure
    vor.weights = weights;

    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}


// Lab 3 (fluids)
class Fluid {
public:
    Fluid(int N_particles = 100) : N_particles(N_particles) {

        ot.fluid_volume = 0.4;

        // WW WE WILL intiialise particles in a hgih cluster. The idea is to make them fall

        for(int i =0; i< N_particles; i++){

            double r = 0.2 * sqrt(uniform(engine[0]));

            double theta = 2.0 * M_PI * uniform(engine[0]);

            particles.push_back(Vector(0.5 + r* cos(theta),0.8 + r* sin(theta)));
            velocities.push_back(Vector(0,0));

            ot.vor.weights.push_back(0.0);
        }
        ot.vor.weights.push_back(0.0);
    }

    // Lab 3 : advance the simulation dt in time
    void time_step(double dt) {

        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;

        // TODO Lab 3 : 
        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell centroid, integrate acceleration->velocity and velocity->position

        // Ok first I will sync the particle positions  and optimize

        ot.vor.points = particles;

        ot.optimize();


        // now we'll compute all the physics and integrate

        for(int i = 0; i<N_particles; i++){
            Vector centroid = ot.vor.cells[i].centroid();

            // Now we'll apply the spring force to pull the partile to the center of the cell

            Vector f_Spring(0,0);

            if(ot.vor.cells[i].area() > 1e-10) {

                f_Spring = (centroid -particles[i])/epsilon2;

            }

            Vector acceleration = g +f_Spring;

            velocities[i] = velocities[i]  + acceleration *dt;
            particles[i] = particles[i] + velocities[i]*dt;

            // WEhave the basic floor and wall collision bounds here

            if (particles[i][0]<0.0){
                particles [i][0] = 0.0;

                velocities[i][0] *= -0.5; 
            }

            if (particles[i][0]>1.0){
                particles [i][0] = 1.0;

                velocities[i][0] *= -0.5; 
            }

            if (particles[i][1]<0.0){
                particles [i][1] = 0.0;

                velocities[i][1] *= -0.5; 

            }

            if (particles[i][1]>1.0){
                particles [i][1] = 1.0;

                velocities[i][1] *= -0.5; 
            }



        }



    }

    // just run the full simulation
    void run_simulation() {
        double dt = 0.002;
        for (int i = 0; i < 100; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "test", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied by the fluid
};

// saves a static svg file. The polygon vertices are supposed to be in the range [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon>& polygons, std::string filename, std::string fillcol = "none") {
    FILE* f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000), (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }
    fprintf(f, "</svg>\n");
    fclose(f);
}







int main() {

    Fluid sim(200);

    std::cout<<"Starting fluid sim " << std::endl;

    sim.run_simulation();
    std::cout<<"Simulation finished " << std::endl;

    return 0;
    
}