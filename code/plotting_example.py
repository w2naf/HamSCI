#!/usr/bin/env python
#Including the above line as the first line of the script allows this script to be run
#directly from the command line without first calling python/ipython.

################################################################################
# For a demonstration, this script will calculate and plot a sine function over
# some defined period of time.
# This script demonstrates some numpy, matplotlib, and datetime techniques.
#
# //W2NAF 15 Feb 2014
################################################################################

#Now we import libraries that are not "built-in" to python.
import os               # Provides utilities that help us do os-level operations like create directories
import datetime         # Really awesome module for working with dates and times.

import matplotlib       # Plotting toolkit
matplotlib.use('Agg')   # Anti-grain geometry backend.
                        # This makes it easy to plot things to files without having to connect to X11.
                        # If you want to avoid using X11, you must call matplotlib.use('Agg') BEFORE calling anything that might use pyplot
                        # I don't like X11 because I often run my code in terminal or web server environments that don't have access to it.
import matplotlib.pyplot as plt #Pyplot gives easier acces to Matplotlib.  

import numpy as np      #Numerical python - provides array types and operations

output_dir  = 'output'

# Plotting section #############################################################
try:    # Create the output directory, but fail silently if it already exists
    os.makedirs(output_dir) 
except:
    pass

# Set up the time and cadence of the time vector.
sTime       = datetime.datetime(2014,1,1)
eTime       = datetime.datetime(2014,1,2)
tDelta      = datetime.timedelta(minutes=1)

curr_time   = sTime
times       = [sTime]
while curr_time < eTime:
    curr_time += tDelta
    times.append(curr_time)

# We can define a function right here to handle the sin calculation.
def my_sin(datetimes,period_in_minutes=120.):
    """Calcuate the sin(datetimes) with a period specified in minutes.
    **datetimes**:          list or numpy array of datetimes
    **period_in_minutes** : optional, defaults to 120 minutes
    """
    # The section in triple quotes above is the docstring for the function, which is printed if you call
    # my_sin? at the iPython prompt (after running this routine).  It is also used by programs that automatically
    # create code documentation, such as Sphynx.

    import time # Used to convert time tuples into unix epoch time
                # Note this import is only available within this function.
                # Any imports/variables defined above and outside of this function are available inside of this function.
                # That's why I can call np within the function.
                # Also, imports just load in "namespaces" into memory, not the actual module.  That means you can import
                # a lot and not cause your program to slow down or use up memory.

    freq    = 1./(period_in_minutes*60.)
    xx      = [time.mktime(dt.timetuple()) for dt in datetimes] # We need the time in some simple numerical format for the sin calculation.
                                                                # I'm using a list comprehension to convert to Unix epoch time, which is measured in seconds.

    yy = np.sin(2*np.pi*freq*np.array(xx)) # I need to convert the xx list into a numpy array in order for the IDL-style multiplication to work.

    # The following line will put a 'stop' in your code, much like in IDL.  You need to have ipdb installed (sudo pip install ipdb). Uncomment it to try it out.
    # Note that it is not exactly the same as dropping to a normal python/ipython shell.  However, it is very, very close.  Use the built-in help if you have trouble.
    #import ipdb; ipdb.set_trace()
    return yy


y   = my_sin(times) # Call the function we made.

fig = plt.figure() #Generate a figure.
ax  = fig.add_subplot(111) #Add an axis to the figure.  Here, we are adding a single plot.
ax.plot(times, y) # Plot the data.

#Add titles, axis labels, and rotate the xtick labels so the times don't overlap each other.
ax.set_title('A Sample Sinusoid\n'+sTime.strftime('%d %b %Y %H%M UT - ')+eTime.strftime('%d %b %Y %H%M UT'))
ax.set_ylabel('Amplitude')
ax.set_xlabel('Time [UT]')
plt.xticks(rotation=30)


fig.tight_layout()  #Some things get clipped if you don't do this.

filename    = os.path.join(output_dir,'example_sinusoid.png')   # os.path.join is an operating system-independent method for joining file paths.
                                                                # i.e., Linux/Mac uses '/' separator, while windows uses '\' separator
fig.savefig(filename) #Save the figure to a file.  Filetype is controlled by the extension on the filename.

