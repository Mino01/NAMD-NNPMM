#!/usr/bin/python

"""
Orca input-file read for TorchANI calculation 

! PM3 EnGrad TightSCF
%output PrintLevel Mini Print[ P_Mulliken ] 1 Print[P_AtCharges_M] 1 end
%pointcharges "/ceph/hpc/data/eurohpc-2010pa6095-users/FORCELAB/HYDROXYAPATITE_SIMULATIONS/BISPHOSPHONATES/NAMD-AmberFF/ML-FPPS/L10/Qmmm/0/qmmm_0.input.pntchrg"


%coords
  CTyp xyz
  Charge 0.000000
  Mult 1.000000
  Units Angs
  coords

  N 57.611782 44.197153 58.221391
  C 58.354525 43.070003 58.077894
  C 59.676092 43.503696 58.216183
  N 59.741297 44.826609 58.476960
  C 58.453786 45.250017 58.488426
  O 53.286913 43.822830 58.536440
  P 53.664260 45.209647 58.082551
  O 52.798575 45.758389 56.972253
  O 53.930071 46.168830 59.209402
  C 55.346539 44.954381 57.295220
  O 55.876800 46.243850 57.142196
  C 56.169403 44.225999 58.394022
  P 55.559256 44.151837 55.625876
  O 54.619177 44.930298 54.718877
  O 57.006260 44.349159 55.317798
  O 55.131589 42.723114 55.875544
  end
end

"""

#from __future__ import print_function
import numpy as np
from sys import argv
#from ase.lattice.cubic import Diamond

import ase 
import torch
import torchani

#from ase.io.trajectory import Trajectory
#import ase.io
import socket
import os
import sys

def splitter(alist):
    vec=[]
    for row in alist:
        vec.append(row.split())
    return vec

def write_file(fname,derivative):
    finFile = open(fname + '.result', 'w')
    finFile.write(str(energy.item()*627.509469) + "\n")
    d=derivative
    for i in range(len(elements)):
        finFile.write(str(-627.509469*d[i][0].item()) + " " + str(-627.509469*d[i][1].item())  +
            ' ' + str(str(-627.509469*d[i][2].item()) ) + " " + str(charges[i]) + "\n")
    
    finFile.close()

class OrcainputParser():
    def __init__(self, filename, natoms): 
        self.filename = filename
        self.data =[]
        self.atomnames=[]
        self.elementnumbers = []
        self.crd =[]
        self.natoms = natoms 

    def readinput(self):
        raw=open(self.filename,'r')
        data = raw.read().split("\n")
        print(data)
        self.data = data 
        xcoord=[]
        ycoord=[] 
        zcoord=[]
        newdata=splitter(self.data)
        length = len(newdata)
        startindex=2
        for row in range(2,startindex+natoms):
            print(newdata[row])
            self.atomnames.append(str(newdata[row][0]))     
            xcoord.append(float(newdata[row][1]))
            ycoord.append(float(newdata[row][2]))
            zcoord.append(float(newdata[row][3]))
            self.crd = [xcoord,ycoord,zcoord]

    def get_atomnames(self):
        return self.atomnames 

    def get_elementnumbers(self): 
        #convert atomnames to atomindex according to periodic table
        #using a dictionary of ani2x elements 
        ani_atomnames = ['H', 'C', 'N', 'O', 'F', 'Cl', 'S']
        atomindex = [1, 6, 7, 8, 9, 17, 16]
        anidict = dict(zip(ani_atomnames,atomindex))
        for item in self.atomnames: 
            self.elementnumbers.append(anidict[item])

        return self.elementnumbers  

    def get_coordinates(self):
        return self.crd

    def printfilename(self):
        print(self.filename)


class TorchaniModel():
    def __init__(self, modelname, elementnumbers, coordinates): 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.species = torch.tensor([elementnumbers], device=self.device)
        self.coordinates = torch.tensor([coordinates], requires_grad=True, device=self.device)
        self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        self.torchenergy = []
        self.torchforce = []
        self.torchderivative =[]
        self.energy = []
        self.derivative =[]
        self.force = []

    def get_energy(self): 
        self.torchenergy = ani2x((species, coordinates)).energies
        self.energy = self.torchenergy.item()
        print('Energy:', self.energy)
        return self.energy 

    def get_derivative(self):
        self.torchderivative = torch.autograd.grad(energy.sum(), coordinates)[0]
        self.derivative = self.torchderivative.squeeze()
        print('Force:', self.derivative.squeeze())
        return self.derivative 

    def get_force(self):
        self.torchderivative = torch.autograd.grad(energy.sum(), coordinates)[0]
        self.torchforce = -self.torchderivative
        self.force = -self.torchderivative.squeeze()
        return self.force

# class NAMDinterface_server():
#     def __init__(self): 
#         sock_address = "/tmp/ani_socket"
#         server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
#         server.bind(sock_address)
#         server.listen(100)

#         while True:
#             conn, _ = server.accept()
#             datagram = conn.recv(1024)
#         if not datagram:
#             break
#         else:


####### MAIN ###########
natoms = 3 
orca = OrcainputParser('water.inp',natoms)
orca.readinput()

print("Read ORCA input file")

print("atomnames:")
atomnames = orca.get_atomnames()
print(atomnames)

print("elementnumbers:")
elementnumbers = orca.get_elementnumbers()
print(elementnumbers)

print("coordinates:")
coordinates = orca.get_coordinates()
print(coordinates)

torchmodel = TorchaniModel('ani2x',elementnumbers,coordinates)

#### Compile and save ANI2x model ###
#compiled_model = torch.jit.script(model)
#torch.jit.save(compiled_model, 'compiled_model.pt')




