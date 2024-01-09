# Access entire backend
import iterativeeikonal.derivativesR2 as derivativesR2
import iterativeeikonal.derivativesSE2 as derivativesSE2
import iterativeeikonal.cleanarrays as cleanarrays
import iterativeeikonal.costfunctions as costfunctions
import iterativeeikonal.solvers as solvers

# Most important functions are available at top level
## R2
from iterativeeikonal.solvers import eikonal_solver_R2, geodesic_back_tracking_R2, convert_continuous_indices_to_real_space_R2
from iterativeeikonal.costfunctions import multiscale_frangi_filter_R2, cost_function
## SE2
from iterativeeikonal.solvers import eikonal_solver_SE2_LI, geodesic_back_tracking_SE2, convert_continuous_indices_to_real_space_SE2
