import numpy as np
import matplotlib.pyplot as plt


def test_Omega(rr):
    return -100*np.sin(2*np.pi*rr)


if __name__=='__main__':

    profile_r = [-0.1, 0.159, 0.22908333333333333, 0.2991666666666667, 0.36924999999999997, 0.43933333333333335, 0.5094166666666666, 0.5795, 0.6495833333333333, 0.7196666666666667, 0.78975, 0.8598333333333333, 0.9299166666666667, 1.1]
    prof = [
(0,0,0),
(-21.517318344634447, 156.9144334583121, 161.11756359899576),
(-57.86983263535715, 81.58373271349082, 79.47135518340136),
(-78.26919033105997, 62.12798118899774, 60.79079409546341),
(-69.96251409442641, 97.05943377820319, 98.57612486429076),
(-99.35512484233735, 88.91816299313714, 88.28622621890815),
(-97.6089406556674, 111.04514926307621, 109.6529681627417),
(-111.92526512018445, 112.90101752207978, 114.13750898813377),
(-59.989818569007184, 153.45927608036956, 152.20615202703164),
(-100.23090221505679, 123.8708986506127, 124.31311387504591),
(20.112149114882797, 113.08539548413337, 117.28632663637914),
(55.579505271951206, 73.6378565688324, 76.70436526222528),
(73.2833757362081, 95.6596455498891, 96.5910198946401)]

  



    
    kern_x = np.linspace(0,1,1000)
    bool_list = []
    for i in range(1, len(profile_r)):
        bool_list.append( (profile_r[i-1] <= kern_x) & (kern_x < profile_r[i]) )

    prof_mid = np.piecewise(kern_x, bool_list, [p[0] for p in prof])
    prof_top = np.piecewise(kern_x, bool_list, [p[0]+p[1] for p in prof])
    prof_bot = np.piecewise(kern_x, bool_list, [p[0]-p[2] for p in prof])

    plt.fill_between(kern_x, prof_bot, prof_top, label='MCMC uncert')
    plt.plot(kern_x, prof_mid, color='orange', label='MCMC')
    plt.plot(kern_x, test_Omega(kern_x), color='black', linewidth=2, label='True')
    plt.xlabel('Radius [R_star]')
    plt.ylabel('Rotation rate [nHz]')
    plt.legend()
    #plt.grid()
    plt.show()





