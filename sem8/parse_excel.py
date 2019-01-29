import numpy as np
import pandas as pd
class Parser:
    def __init__(self):
        self.EXCEL_HEADERS = [0,1,2,3,4]
        self.GAME_FEATS_ST = 8
        self.GAME_FEATS_END = 28
        self.CODE_FEATS_COLS = [0,29,30,31,32,33,35,36,37,38,39,41,42,43,44,45,46]
        self.GRADE_COLS = [6,7]
        self.GRADE_MAP = {'Ex':10,"EX":10,"A":9,"B":8,"C":7,"D":6,"P":5,"F":0}
        self.CGPA_under = 5
        self.CGPA_rs = 6
        self.EXPERTISE_LIMITS = (9.5,8.0)
        self.EXPERTISE_CG_COL = (7,8)
        pass

    def read_all_data(self,file):
        return np.array(pd.read_excel(file,header=self.EXCEL_HEADERS).reset_index())

    def get_game_features(self,data):
        data = data[:,self.GAME_FEATS_ST:self.GAME_FEATS_END]
        X_game = []
        for r in data:
            if not r[0] != r[0]:
                X_game.append(r)
        return np.array(X_game,dtype=np.float64)

    def clean_code_features(self,data):
        for r in data:
            time_indices = [5, 10, 16]
            for j in time_indices:
                if r[j] != r[j]:
                    continue
                if 'sec' in str(r[j]):
                    r[j] = str(r[j])[:-3]
                else:
                    mi = int(r[j])
                    sec = int((float(r[j])-mi)*100)
                    r[j] = mi*60+sec
        return data

    def get_code_features(self,data):
        data = data[:,self.CODE_FEATS_COLS]
        NUM_FEATS = 6
        data = self.clean_code_features(data)
        X_code = []
        ques_num = 0
        for r in data:
            if not r[0] != r[0]:
                ques_num = 0
                X_code.append(np.zeros(5*NUM_FEATS))
            if not r[1] != r[1]:
                temp = np.array(r[1:6],dtype=np.float64)
                #feats = [temp[1]/temp[0],1./(temp[4]/(temp[3]*temp[2]*1.))]
                #X_code[len(X_code)-1][ques_num*NUM_FEATS:ques_num*NUM_FEATS+2] = feats
                X_code[len(X_code)-1][ques_num*6:ques_num*6+5] = r[1:6]
                ques_num += 1
            if not r[6] != r[6]:
                temp = np.array(r[6:11],dtype=np.float64)
                #feats = [temp[1]/temp[0],1./(temp[4]/(temp[3]*temp[2]*1.))]
                #X_code[len(X_code)-1][ques_num*NUM_FEATS:ques_num*NUM_FEATS+2] = feats
                X_code[len(X_code)-1][ques_num*6:ques_num*6+5] = r[6:11]
                ques_num += 1
            if not r[11] != r[11]:
                temp = r[11:13]
                temp = np.array(np.concatenate((temp,r[14:17])),dtype=np.float64)
                #feats = [temp[1]/temp[0],1./(temp[4]/(temp[3]*temp[2]*1.))]
                #X_code[len(X_code)-1][ques_num*NUM_FEATS:ques_num*NUM_FEATS+2] = feats
                X_code[len(X_code)-1][ques_num*6:ques_num*6+2] = r[11:13]
                X_code[len(X_code)-1][ques_num*6+2:ques_num*6+5] = r[14:17]
                if r[13]=='y':
                    #X_code[len(X_code)-1][ques_num*6+5] = 1
                    X_code[len(X_code)-1][ques_num*NUM_FEATS+2] = 1
                ques_num += 1
        return np.array(X_code)

    def parse_grades(self,data,limit):
        data = data[:,self.GRADE_COLS]
        X_grade = []
        for r in data:
            if len(X_grade) == limit:
                break
            if r[0] != r[0]:
                continue
            X_grade.append([self.GRADE_MAP[r[0]],self.GRADE_MAP[r[1]]])

        return np.array(X_grade)

    def distribute_grades(self,data):
        dist = []
        for r in data:
            sum = r[0]+r[1]
            if sum >= 19:
                dist.append(0)
            elif sum >=16:
                dist.append(1)
            else:
                dist.append(2)
        return np.array(dist)

    def parse_cgpa(self,data,limit):
        X_cg = []
        for r in data:
            if r[0] != r[0]:
                continue
            if len(X_cg) >= limit:
                ##RS
                X_cg.append(r[self.CGPA_rs])
            else:
                ##Undergrad
                X_cg.append(r[self.CGPA_under])

        return np.array(X_cg)

    def distribute_cgs(self,data):
        dist = []
        for r in data:
            if r >= 9.0:
                dist.append(0)
            elif r >= 8.0:
                dist.append(1)
            else:
                dist.append(2)
        return np.array(dist)

    def get_expertise_CG(self,data,limit):
        tot = 0
        names = []
        expertise = []
        for r in data:
            if r[0] != r[0]:
                continue
            if tot >= limit:
                cg = r[self.EXPERTISE_CG_COL[1]]
            else:
                cg = r[self.EXPERTISE_CG_COL[0]]
            names.append(r[0].lower())
            if cg>=self.EXPERTISE_LIMITS[0]:
                expertise.append(0)
            elif cg>=self.EXPERTISE_LIMITS[1]:
                expertise.append(1)
            else:
                expertise.append(2)
            tot += 1

        return names,expertise
