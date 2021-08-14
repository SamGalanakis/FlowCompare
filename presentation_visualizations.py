import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import matplotlib as mpl


sns.set_style("darkgrid")


def f(a,b):
     return 1/(b-a)


a=0
b =3


rectangle = mpl.patches.Rectangle((a,0),b-a,f(a,b),color='steelblue')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(left=-1.,right=7.)
ax.add_patch(rectangle)
plt.title('Uniform 0-3')
plt.savefig("Uniform_original.svg")

ax.clear()
rectangle = mpl.patches.Rectangle((1,0),3,f(1,4),color='steelblue')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(left=-1.,right=7.)
ax.add_patch(rectangle)
plt.title('Uniform translation')
plt.savefig("Uniform_translation.svg")


ax.clear()
rectangle = mpl.patches.Rectangle((1,0),6,f(1,7),color='steelblue')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(left=-1.,right=7.)
ax.add_patch(rectangle)
plt.title('Uniform affine')
plt.savefig("Uniform_affine.svg")

