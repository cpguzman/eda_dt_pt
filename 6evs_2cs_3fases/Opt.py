#**************************Import packages******************************
import pyomo
import pyomo.opt
import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib as plt


def _auxDictionary(a):
    temp_dictionary = {}
    if len(a.shape) == 3:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                for dim2 in np.arange(a.shape[2]):
                    temp_dictionary[(dim0+1, dim1+1, dim2+1)] = a[dim0, dim1, dim2]
    elif len(a.shape) == 2:
        for dim0 in np.arange(a.shape[0]):
            for dim1 in np.arange(a.shape[1]):
                temp_dictionary[(dim0+1, dim1+1)] = a[dim0, dim1]
    else:
        for dim0 in np.arange(a.shape[0]):
            temp_dictionary[(dim0+1)] = a[dim0]
    return temp_dictionary
#temp_dict1 = _auxDictionary(loadLimit)

#**************************************Data definition******************************************
data = {}
data['energy_price'] = pd.read_csv('energy_price.csv')
data['evs_inputs'] = pd.read_csv('evs_inputs.csv')
data['alpha'] = pd.read_csv('alpha.csv')
data['css_inputs'] = pd.read_csv('css_inputs.csv')
data['S'] = pd.read_csv('s.csv')
data['cp_inputs'] = pd.read_csv('cp_inputs.csv')
data['fases'] = pd.read_csv('fases.csv')
data['pl'] = pd.read_csv('pl.csv')
data['pt'] = pd.read_csv('pt.csv')
data['pv'] = pd.read_csv('pv.csv')
data['css_power'] = pd.read_csv('css_power.csv')
n_time = data['energy_price']['dT'].size
n_evs = data['evs_inputs']['Esoc'].size
cp = data['cp_inputs']['cs_id'].size
css = data['css_inputs']['cs_id'].size
fases = data['fases']['line'].size

print(n_evs, cp, css, fases)
print(cp)


#***************************************Star time definition********************
from datetime import datetime
now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

#***************************************Sets definition********************
model = pyo.ConcreteModel()
model.ev = pyo.Set(initialize = np.arange(1, n_evs + 1))
model.t = pyo.Set(initialize = np.arange(1, n_time + 1))
model.cs = pyo.Set(initialize = np.arange(1, css + 1))
model.cp = pyo.Set(initialize = np.arange(1, cp + 1))
model.f = pyo.Set(initialize = np.arange(1, fases + 1))

#***************************************Parameters definition********************
model.pt = pyo.Param(model.f, model.t, initialize =_auxDictionary(data['pt'].to_numpy()))
model.pv = pyo.Param(model.f, model.t, initialize =_auxDictionary(data['pv'].to_numpy()))
model.ev_id = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,10]))
model.cs_id = pyo.Param(model.cs, initialize =_auxDictionary(data['css_inputs'].to_numpy()[:,0]))
model.my_cs_id_cp = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,0]))
model.cp_id = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,10]))
model.csconnected = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,11]))
model.ESoc = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,0]))
model.EEVmin = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,1]))
model.EEVmax = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,2]))
model.Etrip = pyo.Param(model.ev, initialize=_auxDictionary(data['evs_inputs'].to_numpy()[:,3]))
model.pl = pyo.Param(model.f, model.t, initialize =_auxDictionary(data['pl'].to_numpy()))
model.PchmaxEV = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,4]))
model.PdchmaxEV = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,5]))
model.evcheff = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,6])) 
model.evdcheff = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,7])) 
model.cheff = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,1])) 
model.dcheff = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,2])) 
model.cpconnected = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,8])) 
model.Pcpmax = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,3])) 
#model.Pcpmax = pyo.Param(model.f, model.cp, initialize = _auxDictionary(data['cps_power'].to_numpy()))
#model.Pcpdis = pyo.Param(model.cp, initialize =_auxDictionary(data['CSs_inputs'].to_numpy()[:,5])) 
model.type_ = pyo.Param(model.ev, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,9])) 
model.v2gcp = pyo.Param(model.cp, initialize =_auxDictionary(data['cp_inputs'].to_numpy()[:,4])) 
model.v2gev = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,9])) 
model.Pcsmax = pyo.Param(model.cs, initialize =_auxDictionary(data['css_inputs'].to_numpy()[:,1])) #No caso de estiver livre de ficar desbalanceado
#model.Pcsmax = pyo.Param(model.f, model.cs, initialize = _auxDictionary(data['css_power'].to_numpy())) #Para deixar atrelado e que fique equilibrado
#model.cpconnected = pyo.Param(model.ev, initialize =_auxDictionary(data['evs_inputs'].to_numpy()[:,8])) 
x = data['cp_inputs'].to_numpy().astype(int)
model.place = pyo.Param(model.cp, initialize =_auxDictionary(x[:,6])) 

#cs_data = data['CSs_inputs'].to_numpy()
#int_cs_data = cs_data.astype(int)
#print(int_cs_data, int_cs_data[:,10], _auxDictionary(int_cs_data[:,10]))
#model.Pcpmax = pyo.Param(model.cp, initialize =_auxDictionary(int_cs_data[:,4])) 
#model.type_ = pyo.Param(model.cp, initialize =_auxDictionary(int_cs_data[:,10])) 

model.dT = pyo.Param(model.t, initialize =_auxDictionary(data['energy_price'].to_numpy()[:,0]))
model.import_price = pyo.Param(model.t, initialize =_auxDictionary(data['energy_price'].to_numpy()[:,1]))
model.export_price = pyo.Param(model.t, initialize =_auxDictionary(data['energy_price'].to_numpy()[:,2]))
model.S = pyo.Param(model.ev, model.t, initialize = _auxDictionary(data['S'].to_numpy()))
model.alpha = pyo.Param(model.ev, model.t, initialize = _auxDictionary(data['alpha'].to_numpy()))
model.my_cp_fases = pyo.Param(model.cp, initialize=_auxDictionary(data['cp_inputs'].to_numpy()[:,8]))
model.penalty1 = 1000000
model.penalty2 = 1000000
model.penalty3 = 0.6
model.DegCost = 0.10
model.m = pyo.Param(model.ev, initialize=_auxDictionary(data['evs_inputs'].to_numpy()[:,11]))

#***************************************Variables definition********************
model.PEV = pyo.Var(model.ev, model.t, domain = pyo.NonNegativeReals, initialize = 0)
model.PEVdc = pyo.Var(model.ev, model.t, domain = pyo.NonNegativeReals, initialize = 0)
model.EEV = pyo.Var(model.ev, model.t, domain = pyo.Reals, initialize = 0)
model.Etriprelax = pyo.Var(model.ev, model.t, domain = pyo.NonNegativeReals, initialize = 0)
model.Eminsocrelax = pyo.Var(model.ev, model.t, domain = pyo.NonNegativeReals, initialize = 0)
model.Etripn = pyo.Var(model.ev, model.t, domain = pyo.Reals, initialize = 0)
model.a = pyo.Var(model.ev, model.t, domain = pyo.Binary,bounds=(0, 1), initialize=0) #EV charging binary 
model.b = pyo.Var(model.ev, model.t, domain = pyo.Binary,bounds=(0, 1), initialize=0) #EV discharging binary 
model.PCP = pyo.Var(model.cp, model.t, domain = pyo.Reals, initialize = 0)
model.PCPdc = pyo.Var(model.cp, model.t, domain = pyo.Reals, initialize = 0)
model.PCS = pyo.Var(model.cs, model.t, domain = pyo.Reals, initialize = 0)
model.grid_import = pyo.Var(model.f,model.t, domain=pyo.NonNegativeReals, initialize = 0)
model.grid_export = pyo.Var(model.f,model.t, domain=pyo.NonNegativeReals, initialize = 0)
model.is_importing = pyo.Var(model.t, domain=pyo.Binary, bounds=(0, 1), initialize=0)
model.is_exporting = pyo.Var(model.t, domain=pyo.Binary, bounds=(0, 1), initialize=0)
model.import_relax = pyo.Var(model.f,model.t, domain=pyo.NonNegativeReals, initialize = 0)
model.export_relax = pyo.Var(model.f,model.t, domain=pyo.NonNegativeReals, initialize = 0)

#****************************************************EV constraints******************************************************
# EV power consumption constraints 
def _power_charging_limit1(m,ev,t): 
    return m.PEV[ev,t] >= 0
model.power_charging_limit1 = pyo.Constraint(model.ev, model.t, rule = _power_charging_limit1)

# EV power consumption constraints
def _power_charging_limit2(m,ev,t): 
    return m.PEV[ev,t] <= m.PchmaxEV[ev]*m.alpha[ev,t]*m.a[ev,t]
model.power_charging_limit2 = pyo.Constraint(model.ev, model.t, rule = _power_charging_limit2)

def _power_discharging_limit1(m,ev,t):
    return m.PEVdc[ev,t] >= 0
model.power_discharging_limit1 = pyo.Constraint(model.ev, model.t, rule = _power_discharging_limit1)

# EV power discharging constraints 
def _power_discharging_limit2(m,ev,t): 
    return m.PEVdc[ev,t] <= m.PdchmaxEV[ev]*m.alpha[ev,t]*m.b[ev,t]*m.v2gev[ev] 
model.power_discharging_limit2 = pyo.Constraint(model.ev, model.t, rule = _power_discharging_limit2)

# EV charging and discharging binary limitation 
def _charging_discharging(m,ev,t): 
    return m.a[ev,t] + m.b[ev,t] <= 1 
model.charging_discharging = pyo.Constraint(model.ev, model.t, rule = _charging_discharging)

# EV energy trip 
#def _balance_etripn(m,ev,t): 
#    return m.Etripn[ev,t] == m.Etrip[ev]*m.S[ev,t]/(sum([m.S[ev,k] for k in np.arange(1, n_time + 1)]))
#model.balance_etripn = pyo.Constraint(model.ev, model.t, rule = _balance_etripn)

# EV energy balance at time 0 
def _balance_energy_EVS(m,ev,t,cp): 
    if t == 1:
        #return m.EEV[ev,t] - m.Etriprelax[ev,t] == m.ESoc[ev] + m.PEV[ev,t]*m.dT[t]*(m.evcheff[ev]*m.cheff[cp]) - m.PEVdc[ev,t]*m.dT[t]/(m.evcheff[ev]*m.cheff[cp]) - m.Etripn[ev,t]
        return m.EEV[ev,t] == m.ESoc[ev] + m.PEV[ev,t]*m.dT[t]*m.evcheff[ev] - m.PEVdc[ev,t]*m.dT[t]/(m.evcheff[ev]*m.cheff[cp]) 
    elif t > 1:
        #return m.EEV[ev,t] - m.Etriprelax[ev,t] == m.EEV[ev,t-1] + m.PEV[ev,t]*m.dT[t]*m.evcheff[ev] - m.PEVdc[ev,t]*m.dT[t]/m.evcheff[ev] - m.Etripn[ev,t]
        return m.EEV[ev,t] == m.EEV[ev,t-1] + m.PEV[ev,t]*m.dT[t]*m.evcheff[ev] - m.PEVdc[ev,t]*m.dT[t]/m.evcheff[ev]
model.balance_energy_EVS = pyo.Constraint(model.ev, model.t, model.cp, rule = _balance_energy_EVS)

# EV minimum capacity limitation
def _energy_limits_EVS_1(m,ev,t): 
    return m.EEV[ev,t] + m.Eminsocrelax[ev,t] >= m.EEVmin[ev]
    #return m.EEV[ev,t] >= m.EEVmin[ev]
model.energy_limits_EVS_1 = pyo.Constraint(model.ev, model.t, rule = _energy_limits_EVS_1)

# EV maximum capacity limitation
def _energy_limits_EVS_2(m,ev,t): 
    return m.EEV[ev,t] <= m.EEVmax[ev] 
model.energy_limits_EVS_2 = pyo.Constraint(model.ev, model.t, rule = _energy_limits_EVS_2)  

#Target 
def _balance_energy_EVS3(m,ev,t): 
    if t == 24: #Isto tem que ver a hora de partida do carro (?) no lugar do 24, ou seja vamos por na bateria um target para quando ele deixar o parque
        #return m.EEV[ev,t] + m.Etriprelax[ev,t] >= m.EEVmax[ev]*m.m[ev]
        return m.EEV[ev,t] >= m.EEVmax[ev]*m.m[ev]
    return pyo.Constraint.Skip
model.balance_energy_EVS3 = pyo.Constraint(model.ev, model.t, rule = _balance_energy_EVS3)

#****************************************************CP constraints******************************************************

def _cp_power_charging_limit(m,ev,t,cp): 
#Type_cp indiques if the CP is controlable or not (on-off socket). It is considered that the ev users have an id and this is correlated with the localization 
#into the CSs
    if m.type_[cp] == 1 and m.place[cp] == m.ev_id[ev]:  
        #print(f"Entrou no IF no type 1 com ev = {ev}, t = {t}, cp = {cp}, m.type_[ev] = {m.type_[cp]} m.Pcpmax[cp] = {m.Pcpmax[cp]} m.place[cp] = {m.place[cp]}")
        return m.PEV[ev,t] <= m.Pcpmax[cp]*m.alpha[ev,t]*m.a[ev,t]
    elif m.type_[cp] == 2 and m.place[cp] == m.ev_id[ev]:   
        #print(f"Entrou no ELSE no type 2 com ev = {ev}, t = {t}, cp = {cp},  m.type_[ev] = {m.type_[cp]} m.Pcpmax[cp] = {m.Pcpmax[cp]} m.place[cp] = {m.place[cp]}")
        return m.PEV[ev,t] == m.Pcpmax[cp]*m.alpha[ev,t]*m.a[ev,t]
    return pyo.Constraint.Skip
model.cp_power_charge_limit = pyo.Constraint(model.ev, model.t, model.cp, rule = _cp_power_charging_limit)


def _cp_power_discharging_limit(m,ev,t,cp): 
    if m.type_[cp] == 1 and m.place[cp] == m.ev_id[ev]: 
        #print(f"Entrou no IF no type {m.type} com ev = {ev}, t = {t}, cp = {cp}")
        return m.PEVdc[ev,t] <= m.Pcpmax[cp]*m.alpha[ev,t]*m.b[ev,t]*m.v2gcp[cp]
    elif m.type_[cp] == 2 and m.place[cp] == m.ev_id[ev]: 
        #print(f"Entrou no ELSE no type {m.type} c#Energy balance in the system considering threephase balanced PV, threephase unbalanced load consumption, and threphase unbalanced CS om ev = {ev}, t = {t}, cp = {cp}")
        return m.PEVdc[ev,t] == 0
    #else:
        #print(f"bnana {m.type} com ev = {ev}, t = {t}, cp = {cp}")
        #return 0 
    return pyo.Constraint.Skip      
model.cp_power_discharge_limit = pyo.Constraint(model.ev, model.t, model.cp, rule = _cp_power_discharging_limit)

#Auxiliar expression to obtain the power consumption of each CP related to each EV consumption connected to its
def _cp_power_consumption(m,ev,t,cp): 
    if m.type_[cp] == 1 and m.place[cp] == m.ev_id[ev]:  
       return m.PCP[cp,t] == m.PEV[ev,t]
    elif m.type_[cp] == 2 and m.place[cp] == m.ev_id[ev]:   
        return m.PCP[cp,t]== m.PEV[ev,t]
    return pyo.Constraint.Skip
model._cp_power_consumption = pyo.Constraint(model.ev, model.t, model.cp, rule = _cp_power_consumption)

#Auxiliar expression to obtain the discharging power of each CP related to each EV discharging connected to its
def _cp_power_discharging(m,ev,t,cp): 
    if m.type_[cp] == 1 and m.place[cp] == m.ev_id[ev]:  
       return m.PCPdc[cp,t] == m.PEVdc[ev,t] 
    elif m.type_[cp] == 2 and m.place[cp] == m.ev_id[ev]:   
        return m.PCPdc[cp,t] == m.PEVdc[ev,t] 
    return pyo.Constraint.Skip
model._cp_power_discharging = pyo.Constraint(model.ev, model.t, model.cp, rule = _cp_power_discharging)

#****************************************************CS constraints******************************************************
#*************** CS power consumption limitation**************************
#def _cs_power_charging_limit(m,f,t,cs,cp): #This is used when the CS power is settled to be equilibrium between the 3 phases
def _cs_power_charging_limit(m,t,cs,cp): #This is used when the CS power is divided by three, and therefore it is possible to be not equilibrium between the 3 phases
    print("CS_ID ->CS", m.cs_id[cs], "CS_ID ->CP", m.my_cs_id_cp[cp], "CS", cs, "CP",cp)
    # (m.cs_id[cs] == m.my_cs_id_cp[cp]) identifies which CPs are connected to a CS
    if m.cs_id[cs] == m.my_cs_id_cp[cp]: 
        print(f"-----------------NEXT ROUND------------------\né a vez do cp {cp}")
        soma = m.PCP[cp,t] - m.PCPdc[cp,t]
        #print(f"cs{cs} cp {cp} m.my_cp_fases[cp] {m.my_cp_fases[cp]}")
        for other_cp in m.cp:
            #print(f"Uepa!! m.my_cs_id_cp[cp] = {m.my_cs_id_cp[cp]} e m.my_cs_id_cp[other_cp] =  ")
            if other_cp != cp and m.my_cp_fases[cp] == m.my_cp_fases[other_cp] and m.my_cs_id_cp[cp] == m.my_cs_id_cp[other_cp]:
                print(f"Entrou Entrou! {other_cp} {cp}")
                soma = soma + m.PCP[other_cp,t] - m.PCPdc[other_cp,t] 
                print(soma)
                
        print(cp, m.cp_id[cp])
        print(soma)
        return soma <= m.Pcsmax[cs]/3 #if it is isolated, the sum will always be just it. If not, it will be the sum of it and the others
  
    return pyo.Constraint.Skip    
#model.cs_power_charging_limit = pyo.Constraint(model.f, model.t, model.cp, model.cs, rule = _cs_power_charging_limit) #This is used when the CS power is settled to be equilibrium between the 3 phases 
model.cs_power_charging_limit = pyo.Constraint(model.t, model.cs,model.cp, rule = _cs_power_charging_limit) #This is used when the CS power is divided by three, and therefore it is possible to be not equilibrium between the 3 phases

#*************** CS power discharging limitation**************************
#def _cs_power_discharging_limit(m,f,t,cs,cp): This is used when the CS power is settled to be equilibrium between the 3 phases
def _cs_power_discharging_limit(m,t,cs,cp): #This is used when the CS power is divided by three, and therefore it is possible to be not equilibrium between the 3 phases
    if m.cs_id[cs] == m.my_cs_id_cp[cp]: 
        print(f"-----------------NEXT ROUND-------------------\né a vez do cp {cp}")
        
        soma = m.PCP[cp,t] - m.PCPdc[cp,t]
        print(f"cs{cs} cp {cp} m.my_cp_fases[cp] {m.my_cp_fases[cp]}")
        for other_cp in m.cp:
            if other_cp != cp and m.my_cp_fases[cp] == m.my_cp_fases[other_cp] and m.my_cs_id_cp[cp] == m.my_cs_id_cp[other_cp]:
                print(f"Entrou Entrou! {other_cp} {cp}")
                soma = soma + m.PCP[other_cp,t] - m.PCPdc[other_cp,t] 
                print(soma)
                
        print(cp, m.cp_id[cp])
        print(soma)
        return soma >= -1 * m.Pcsmax[cs]/3 #if it is isolated, the sum will always be just it. If not, it will be the sum of it and the others
    return pyo.Constraint.Skip
#model.cs_power_discharging_limit = pyo.Constraint(model.f,model.t, model.cs, model.cp,  rule = _cs_power_discharging_limit) #This is used when the CS power is settled to be equilibrium between the 3 phases 
model.cs_power_discharging_limit = pyo.Constraint(model.t, model.cs, model.cp,  rule = _cs_power_discharging_limit) #This is used when the CS power is divided by three, and therefore it is possible to be not equilibrium between the 3 phases

#Auxiliary expression to obtain the power consumption/discharge of the CS related to the CPs that are connected to it
def _cs_power_charge_discharge_limit(m,t, cs): 
    return m.PCS[cs,t]  == sum([m.PCP[cp,t] - m.PCPdc[cp,t] for cp in m.cp if m.cs_id[cs] == m.my_cs_id_cp[cp]])
model._cs_power_charge_discharge_limit = pyo.Constraint(model.t, model.cs, rule =_cs_power_charge_discharge_limit)  


#**********************************************Company constraints************************************
#Energy balance in the system considering threephase balanced PV, threephase unbalanced load consumption, and threphase unbalanced CS 
def _energy_balance(m,f,t): 
    #return m.grid_import[t]  == sum(m.PCS[cs,t] for cs in m.cs)
    return m.grid_import[f,t]  == sum(m.PCS[cs,t]/3 for cs in m.cs) + m.pl[f,t]*5 - m.pv[f,t] + model.grid_export[f,t] 
model._energy_balance = pyo.Constraint(model.f, model.t, rule =_energy_balance)  

#Contracted power limitation
def _contracted_power_constraint(m,f,t): 
    #return m.grid_import[f,t]  <= m.pt[f,t]*m.is_importing[t] + m.import_relax[f,t]
    return m.grid_import[f,t]  <= m.pt[f,t]*m.is_importing[t]
model._contracted_power_constraint = pyo.Constraint(model.f, model.t, rule =_contracted_power_constraint)  

def _contracted_power_constraint2(m,f,t): 
    #return m.grid_export[f,t]  <= m.pt[f,t]*m.is_exporting[t] + m.export_relax[f,t]
    return m.grid_export[f,t]  <= m.pt[f,t]*m.is_exporting[t] 
model._contracted_power_constrain2 = pyo.Constraint(model.f, model.t, rule =_contracted_power_constraint2)  


def _importing_exporting(m,t): 
    return m.is_importing[t] + m.is_exporting[t]  <= 1
model._importing_exporting= pyo.Constraint(model.t, rule =_importing_exporting)  


#************************************************************************Objective Function***********************************************************
def _FOag(m):
    #return sum([m.PEV[ev,t]*m.dT[t]*m.import_price[t] - m.PEVdc[ev, t]*m.dT[t]*(m.import_price[t]- m.DegCost) + (m.Etripn[ev,t] - m.EEV[ev,t]) + m.Etriprelax[ev,t]*m.penalty1 + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for t in np.arange(1, n_time + 1)])
    #return sum(m.grid_import[f,t] *(m.import_price[t]) +  (m.Etripn[ev,t] - m.EEV[ev,t])*0.1 + m.Etriprelax[ev,t]*m.penalty1 + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for f in np.arange(1, fases + 1) for t in np.arange(1, n_time + 1)) 
    #return sum(m.grid_import[f,t] *(m.import_price[t]) - m.grid_export[f,t] *(m.export_price[t]) + (m.EEVmax[ev] - m.EEV[ev,t])*0.1 + m.import_relax[f,t]*0.1 + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for f in np.arange(1, fases + 1) for t in np.arange(1, n_time + 1))     
    return sum(m.grid_import[f,t]*m.dT[t] *(m.import_price[t]) - m.grid_export[f,t]*m.dT[t]*(m.export_price[t]) + (m.EEVmax[ev] - m.EEV[ev,t])*0.1  + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for f in np.arange(1, fases + 1) for t in np.arange(1, n_time + 1))  
    #return sum(m.grid_import[f,t] *(m.import_price[t]) - m.grid_export[f,t]*(m.export_price[t])  + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for f in np.arange(1, fases + 1) for t in np.arange(1, n_time + 1))  
model.FOag = pyo.Objective(rule = _FOag, sense = pyo.minimize)

#************************************************************************Solve the model***********************************************************
from pyomo.opt import SolverFactory
model.write('res_V4_EC.lp',  io_options={'symbolic_solver_labels': True})

opt = pyo.SolverFactory('cplex', executable='C:/Program Files/IBM/ILOG/CPLEX_Studio129/cplex/bin/x64_win64/cplex.exe')
opt.options['LogFile'] = 'res_V4_EC.log'

results = opt.solve(model)#, tee=True)
results.write()

#************************************************************************End Time information***********************************************************
pyo.value(model.FOag)

now = datetime.now()

end_time = now.strftime("%H:%M:%S")
print("End Time =", end_time)
print("Dif: {}".format(datetime.strptime(end_time, "%H:%M:%S") - datetime.strptime(start_time, "%H:%M:%S")))


def ext_pyomo_vals(vals):
    # make a pd.Series from each
    s = pd.Series(vals.extract_values(),
                  index=vals.extract_values().keys())
    # if the series is multi-indexed we need to unstack it...
    if type(s.index[0]) == tuple:    # it is multi-indexed
        s = s.unstack(level=1)
    else:
        # force transition from Series -> df
        s = pd.DataFrame(s)
    return s

PEV_df = ext_pyomo_vals(model.PEV)
PCP_df = ext_pyomo_vals(model.PCP)
PCS_df = ext_pyomo_vals(model.PCS)
dT_df = ext_pyomo_vals(model.dT)
import_price_df = ext_pyomo_vals(model.import_price)
PEVdc_df = ext_pyomo_vals(model.PEVdc)
EEV_df = ext_pyomo_vals(model.EEV)
grid_import_df = ext_pyomo_vals(model.grid_import) 
grid_export_df = ext_pyomo_vals(model.grid_export)
EEVmax_df = ext_pyomo_vals(model.EEVmax)
Etriprelax_df = ext_pyomo_vals(model.Etriprelax)
Eminsocrelax_df = ext_pyomo_vals(model.Eminsocrelax)
Etripn_df = ext_pyomo_vals(model.Etripn)

#return sum(m.grid_import[f,t] *(m.import_price[t]) - m.grid_export[f,t]*(m.export_price[t]) + (m.EEVmax[ev] - m.EEV[ev,t])*0.1  + m.Eminsocrelax[ev,t]*m.penalty2 for ev in np.arange(1, n_evs + 1) for f in np.arange(1, fases + 1) for t in np.arange(1, n_time + 1)) 

second_term = sum([(EEVmax_df - EEV_df[0][t])*0.1  for t in np.arange(1, n_time + 1)])

charge_cost = sum([PEV_df[t][ev]*dT_df[0][t]*import_price_df[0][t]
                   for ev in np.arange(1, n_evs + 1) for t in np.arange(1, n_time + 1)])

discharge_cost = sum([PEVdc_df[t][ev]*dT_df[0][t]*import_price_df[0][t]
                      for ev in np.arange(1, n_evs + 1) for t in np.arange(1, n_time + 1)])

print('Charge cost: {}'.format(charge_cost))
print('Discharge cost: {}'.format(discharge_cost))

print("Total Charge: {}".format(np.sum(PEV_df.to_numpy())))
print("Total Discharge: {}".format(np.sum(PEVdc_df.to_numpy())))




import os 
folder = 'RESULTS_' + str(n_evs)

if not os.path.exists(folder):
    os.makedirs(folder)
    
EEV_df.to_csv(folder + '/EEV.csv')
EEVmax_df.to_csv(folder + '/EEVmax.csv')
PEV_df.to_csv(folder + '/PEV.csv')
PCP_df.to_csv(folder + '/PCP.csv')
PCS_df.to_csv(folder + '/PCS.csv')
grid_import_df.to_csv(folder + '/grid_import.csv')
grid_export_df.to_csv(folder + '/grid_export.csv')


PEVdc_df.to_csv(folder + '/PEVdc.csv')
PEV_df.sum().to_csv(folder + '/PEV_h.csv')
PEVdc_df.sum().to_csv(folder + '/PEVdc_h.csv')
grid_import_df.sum().to_csv(folder + '/grid_import_h.csv')
grid_export_df.sum().to_csv(folder + '/grid_export_h.csv')

Etriprelax_df.to_csv(folder + '/Etriprelax.csv')
Etriprelax_df.sum().to_csv(folder + '/Etriprelax_h.csv')

Eminsocrelax_df.to_csv(folder + '/Eminsocrelax.csv')
Eminsocrelax_df.sum().to_csv(folder + '/Eminsocrelax_h.csv')

Etripn_df.to_csv(folder + '/Etripn.csv')
Etripn_df.sum().to_csv(folder + '/Etripn_h.csv')