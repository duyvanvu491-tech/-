import numpy as np
class Geometry:
  def __init__(self):
        self.parts = {}      
        self.contacts = {}   
        self.names = []
  def load(self,path):
      self.parts={}
      verts=[]
      curr=None
      with open(path,'r') as f:
          for line in f:
              row=line.split()
              if len(row)==0:
                  continue
              if row[0] == 'v':
                   x_text = row[1] 
                   y_text = row[2]
                   z_text = row[3]
                   x_num = float(x_text)
                   y_num = float(y_text)
                   z_num = float(z_text)
                   point = [x_num, y_num, z_num]
                   verts.append(point)
              elif row[0] == 'g':
                  curr = row[1]
                  new_part_info = {}
                  new_part_info['faces']=[]
                  new_part_info['s']=0
                  self.parts[curr] = new_part_info
              elif row[0] == 'f' and curr:
                   face = [] 
                   for i in row[1:]: 
                    index = int(i) - 1   
                    point = verts[index] 
                    face.append(point)   
                   self.parts[curr]['faces'].append(face)
          self.names = sorted(list(self.parts.keys()))
          self.calc_S()
          self.calc_Sij()
  def get_area(self, pts):
        if len(pts) < 3: 
            return 0.0
        v0 = np.array(pts[0])
        s = 0.0
        for i in range(1, len(pts)-1):
            v1 = np.array(pts[i])
            v2 = np.array(pts[i+1])
            s += 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0))
        return s
  def calc_S(self):
    for data in self.parts.values():
        data['S'] = sum(self.get_area(face) for face in data['faces'])
  def calc_Sij(self):
        self.contacts = {}
        face_map = {}
        
        for name, data in self.parts.items():
            for face in data['faces']:
                rounded_face = []
                for v in face:
                    rounded_face.append(tuple(np.round(v, 4)))
                key = tuple(sorted(rounded_face))
                
                if key not in face_map:
                    face_map[key] = []
                face_map[key].append(name)

        for key, names in face_map.items():
            unique_names = sorted(list(set(names)))
            if len(unique_names) == 2:
                area = self.get_area(key)
                pair = (unique_names[0], unique_names[1])
                self.contacts[pair] = self.contacts.get(pair, 0.0) + area