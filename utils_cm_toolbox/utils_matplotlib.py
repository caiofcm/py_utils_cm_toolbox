############################################
###########################################
#
#
# 	 FIGURE SETTTINGS
#
#
###########################################
###########################################
from matplotlib import pyplot as plt
import inspect
import datetime
import os

# def saveFigureToFolder(name, ext='png', saveKeywords={}, subfolder='figures', fileroot=None):
# 	if fileroot is None:
# 		# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# 		FILE_DIR = os.getcwd()
# 	else:
# 		FILE_DIR = fileroot
# 	FIG_PREFIX = FILE_DIR + r'./' + subfolder + '/' + \
# 		datetime.datetime.now().strftime("%Y-%m-%d") + '-'
# 	os.makedirs(os.path.dirname(FILE_DIR + r'./figures/'), exist_ok=True)
# 	plt.savefig(FIG_PREFIX + name + '.' + ext, **saveKeywords)
# 	return FIG_PREFIX


def saveFiguresToDirectory(	dirName, listFilename=None, lstFigNumber=None,
                            includeDate=False, **svfig_kw):
	"""
	Save figures to file. (more generic than the former function)
	"""
	print(os.getcwd())
	if 'format' in svfig_kw:
		fmrt = svfig_kw['format']
	else:
		fmrt = 'png'
	if listFilename is None:
		if lstFigNumber is None:  # Plot all
			numfigs_active = plt.get_fignums()
			listFilename = ['plot-num-{:d}.{}'.format(k, fmrt) for k in numfigs_active]
			lstFigNumber = numfigs_active
		else:
			listFilename = [
				'plot-num-{:d}.{}'.format(k, fmrt) for k in range(0, len(lstFigNumber))]

	if includeDate:
		date = datetime.datetime.now().strftime("%Y-%m-%d") + '-'
		listFilename = [date + name for name in listFilename]

	#fld = os.path.join(os.path.dirname(__file__), dirName)
	fld = os.path.join(os.getcwd(), dirName)
	if not os.path.isdir(fld):
		os.makedirs(fld)
	for k, fname in enumerate(listFilename):
		plt.figure(lstFigNumber[k])
		plt.savefig(fld + r'/' + fname, **svfig_kw)


def modifyMatplotlibRCParametersForScientifcPlot():
	plt.style.use(['grayscale', 'seaborn-white', 'seaborn-paper'])
	import matplotlib
	# matplotlib.rc('font', **{'size': 14})
	# matplotlib.rcParams.update({'font.size': 22})
	matplotlib.rc('xtick', labelsize=20)
	matplotlib.rc('ytick', labelsize=20)
	matplotlib.rc('legend', fontsize=16)
	matplotlib.rc('axes', labelsize=22)
	# FILE_DIR = os.path.dirname(os.path.abspath(__file__))
