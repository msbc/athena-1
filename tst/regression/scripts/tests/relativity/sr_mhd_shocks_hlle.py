# Test script for relativistic MHD shock tubes with HLLE

# Modules
import numpy as np
import scripts.utils.athena as athena
import scripts.utils.comparison as comparison

# Prepare Athena++
def prepare():
  athena.configure('bs',
      prob='shock_tube_rel',
      coord='cartesian')
  athena.make()

# Run Athena++
def run():
  arguments = [
      'job/problem_id=',
      'output1/file_type=vtk',
      'output1/variable=cons',
      'output1/dt=',
      'time/tlim=',
      'mesh/nx1=']
  times = [0.4, 0.55, 0.5]
  zones = [400, 800, 800]
  for i,time,zone in zip([1,2,4],times,zones):
    arguments_copy = list(arguments)
    arguments_copy[0] += 'sr_mhd_shock' + repr(i)
    arguments_copy[3] += repr(time)
    arguments_copy[4] += repr(time)
    arguments_copy[5] += repr(zone)
    athena.run('mhd_sr/athinput.mub_'+repr(i), arguments_copy)

# Analyze outputs
def analyze():
  headers = [('dens',), ('Etot',), ('mom',0), ('mom',1), ('mom',2),
      ('cell-centered B',0), ('cell-centered B',1), ('cell-centered B',2)]
  tol_sets = [[0.02,  0.02,  0.03,  0.07, 0.0,   0.0, 0.02,  0.0],
              [0.003, 0.002, 0.007, 0.01, 0.007, 0.0, 0.005, 0.007],
              [0.003, 0.002, 0.03,  0.04, 0.01,  0.0, 0.002, 0.005]]
  for i,tols in zip([1,2,4],tol_sets):
    x_ref,_,_,data_ref = athena.read_vtk('data/sr_mhd_shock{0}_hlle.vtk'.format(i))
    x_new,_,_,data_new = \
        athena.read_vtk('bin/sr_mhd_shock{0}.block0.out1.00001.vtk'.format(i))
    for header,tol in zip(headers,tols):
      array_ref = data_ref[header[0]]
      array_ref = array_ref[0,0,:] if len(header) == 1 else array_ref[0,0,:,header[1]]
      array_new = data_new[header[0]]
      array_new = array_new[0,0,:] if len(header) == 1 else array_new[0,0,:,header[1]]
      eps = comparison.l1_diff(x_ref, array_ref, x_new, array_new)
      if tol == 0.0:
        if eps > 0.0:
          return False
      else:
        eps /= comparison.l1_norm(x_ref, array_ref)
        if eps > tol or np.isnan(eps):
          return False
  return True