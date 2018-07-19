import numpy as np
import quaternion
from math import sqrt, atan2, asin, cos, sin
import os
import pickle, scipy.io # libraries used to save kinematics data
import argparse

class TendonDrivenContinuumRobot():

   precision_digits = 16

   def __init__(self):
      self.l1min = 0.085
      self.l1max = 0.115
      self.l2min = 0.185
      self.l2max = 0.215
      self.d = 0.01
      self.n = 10

      # declare a few attributs that can be accessed at any time
      self.base = np.array([0.0, 0.0, 0.0, 1.0]) # base vector used for transformations
      self.l11 = None; self.l12 = None; self.l13 = None; # tendon lengths
      self.l21 = None; self.l22 = None; self.l23 = None; # absolute segment 2 tendon lengths
      self.dl21 = None; self.dl22 = None; self.dl23 = None; # effective  segment 2 tendon lengths (needed for kappa2, phi2, seg_len2)
#      self.lengths = np.zeros(6) # [l11, l12, l13, l21, l22, l23]
      self.lengths1 = np.zeros(3) # [l11, l12, l13]
      self.lengths2 = np.zeros(3) # [l21, l22, l23]
      self.lengths2_effective = np.zeros(3) # [l21-l11, l22-l12, l23-l13]

      self.kappa1 = None # curvature kappa [m^(-1)]
      self.kappa2 = None # curvature kappa [m^(-1)]
      self.phi1 = None # angle rotating arc out of x-z plane [rad]
      self.phi2 = None # angle rotating arc out of x-z plane [rad]
      self.seg_len1 = None # total arc length [m]
      self.seg_len2 = None # total arc length [m]

      self.T01 = None # bishop transformation matrix from base to segment 1 tip
      self.T12 = None # bishop transformation matrix from segment 1 tip to segment 2 tip
      self.T02 = None # bishop transformation matrix from base to segment 2 tip

      self.tip_vec1 = None # segment1 tip vector [m] [x, y, z]
      self.tip_vec2 = None # robot's tip vector [m] [x, y, z]

      self.quat = None # quaternion

   def forward_kinematics(self, tendon_lengths=None, num_poses=1, save_data=False):
      """
      Calculates the tip coordinates/positions [x,y,z] and the orientation quaternion

      tendon_lengths: If
      """
      if tendon_lengths is not None:
         assert np.array(tendon_lengths).shape[1] == 6, "expected array of shape (n, 6) for tendon lengths"
         num_poses = len(tendon_lengths[0]) # get number of given tendon length tuples
         num_poses = 1
      lengths = np.zeros((num_poses, 6))
      positions = np.zeros((num_poses, 3))
      orientations = np.zeros((num_poses, 4))
      print("tendon lengths within forward kinematics", tendon_lengths)
      print("tendon lengths within forward kinematics", tendon_lengths[0])
      for i in range(num_poses):
         if tendon_lengths is None:
            self.lengths1 = np.random.uniform(self.l1min, self.l1max, 3)
            self.lengths2 = np.random.uniform(self.l2min, self.l2max, 3)
         elif len(tendon_lengths[0]) == 6:
#            tendon_lengths = tendon_lendths[0]
            self.lengths1 = tendon_lengths[0][:3]
            print("lengths1: {}".format(self.lengths1))
            assert all(l >= self.l1min and l <= self.l1max for l in self.lengths1), \
               "Given tendon lengths for segment 1 are not within range of l1min and l1max."
            self.lengths2 = tendon_lengths[0][3:6]
            print("lengths2: {}".format(self.lengths2))
            assert all(l >= self.l2min and l <= self.l2max for l in self.lengths2), \
               "Given tendon lengths for segment 2 are not within range of l2min and l2max."
         self.lengths2_effective = self.lengths2-self.lengths1
         self.kappa1, self.phi1, self.seg_len1 = self.arc_params(self.lengths1)
         self.kappa2, self.phi2, self.seg_len2 = self.arc_params(self.lengths2_effective)
         self.T01 = self.transformation_matrix(self.kappa1, self.phi1, self.seg_len1)
         self.T12 = self.transformation_matrix(self.kappa2, self.phi2, self.seg_len2)
         self.T02 = np.matmul(self.T01, self.T12)
         self.tip_vec1 = np.matmul(self.T01, self.base)[0:3]
         self.tip_vec2 = np.matmul(self.T02, self.base)[0:3]
         self.quat = quaternion.as_float_array(quaternion.from_rotation_matrix(self.T02))
         if np.sign(self.quat[0]) != 0.0:
            self.quat *= np.sign(self.quat[0]) # make sure quaternions have the same signum due to R(q) = R(-q)
         lengths[i] = np.concatenate((self.lengths1, self.lengths2))
         positions[i] = self.tip_vec2
         orientations[i] = self.quat

      if save_data:
         self.save_data(lengths, positions, orientations, num_poses)
      return lengths, positions, orientations

   def arc_params(self, lengths):
      """Returns arc parameters kappa, phi, s of continuum robot."""
      l1 = lengths[0]; l2 = lengths[1]; l3 = lengths[2]
      # useful expressions to shorten formulas below
      lsum = l1+l2+l3
      expr = l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3
      # in rare cases expr ~ +-1e-16 when l1~l2~l3 due to floating point operations
      # in these cases expr has to be set to 0.0 in order to handle the singularity
      if round(abs(expr), self.precision_digits) == 0:
         expr = 0.0
      print("expression", expr)
      print("lsum", lsum)
      kappa = 2*sqrt(expr) / (self.d*lsum)
      phi = atan2(sqrt(3)*(l2+l3-2*l1), 3*(l2-l3))
      # calculate total segment length
      if l1 == l2 == l3 or expr == 0.0: # handling the singularity
         seg_len = lsum / 3
      else:
         seg_len = self.n*self.d*lsum / sqrt(expr) * asin(sqrt(expr)/(3*self.n*self.d))
      return kappa, phi, seg_len

   def transformation_matrix(self, kappa, phi, s):
      if round(kappa, self.precision_digits) == 0.0: #handling singularity
         T = np.identity(4)
         T[2, 3] = s
      else:
         T = np.array([[cos(phi)**2*(cos(kappa*s)-1)+1, sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)*sin(kappa*s), cos(phi)*(1-cos(kappa*s))/kappa],
                       [sin(phi)*cos(phi)*(cos(kappa*s)-1), cos(phi)**2*(1-cos(kappa*s))+cos(kappa*s), sin(phi)*sin(kappa*s), sin(phi)*(1-cos(kappa*s))/kappa],
                       [-cos(phi)*sin(kappa*s), -sin(phi)*sin(kappa*s), cos(kappa*s), sin(kappa*s)/kappa],
                       [0, 0, 0, 1]])
      return T

   def save_data(self, lengths, positions, orientations, num_poses):
      directory = os.path.dirname(os.path.abspath(__file__))
      print(directory)
      save_dict = {"lengths": lengths, "positions": positions, "orientations": orientations}
      abspath_pkl = directory + "/data/data" + str(num_poses) + ".pkl"
      abspath_mat = directory + "/data/data" + str(num_poses) + ".mat"
      pickle.dump(save_dict, open(abspath_pkl, "wb"))
      scipy.io.savemat(abspath_mat, mdict=save_dict)
      print("saved {} and {} to {}".format(os.path.basename(abspath_pkl), os.path.basename(abspath_mat), os.path.dirname(os.path.abspath(abspath_mat))))


#parser = argparse.ArgumentParser()
#parser.add_argument("--num_poses", type=int, default=100,
#                    help="Number of poses to be saved to pkl and mat file containing: \
#                          tendon_lengths, positions, quaternions")
#parser.add_argument("--save_data", type=int, default=0,
#                    help="Saves (overwrites) data to .pkl and .mat file if evaluated to true")
#args = parser.parse_args()
#
#TendonDrivenContinuumRobot().forward_kinematics(num_poses=args.num_poses, save_data=args.save_data)