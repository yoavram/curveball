import warnings
import sys
import requests
import json
import os

if __name__ == '__main__':
	base_url = 'https://app.wercker.com'
	try:
		with open('.wercker.token') as f:
			token = f.read()
	except IOError:
		print "Can't find wercker token"
		sys.exit()
	username = 'yoavram'
	app_name = 'curveball'
	app_id = ''

	apps_url = '%s/api/v3/applications/%s' % (base_url, username)
	builds_url = '%s/api/v3/builds' % base_url
	headers = headers = {'Authorization':'Bearer %s' % token}

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		r = requests.get(apps_url, headers=headers, verify=True)
	print "Apps:"
	for app in r.json():
		print app['name'], app['id']
		if app['name'] == app_name:
			app_id = app['id']

	if not app_id:
		print "We have a problem, no app id found!"
		sys.exit()

	# http://devcenter.wercker.com/api/endpoints/builds.html#trigger-a-build
	payload = {
				"applicationId": app_id, 
				#"branch":"master", 
				#"commit":"no-id", 
				#"message":"trigger build",
				"envVars": [{'key':'GITHUB_USERNAME', 'value':'yoavram'},
							{'key':'GITHUB_EMAIL', 'value':'yoavram+github@gmail.com'},
							{'key':'GITHUB_REPO', 'value':'curveball'},
							{'key':'GITHUB_TOKEN', 'value':'token'},
							{'key':'DST_COVERAGE_DIR', 'value':'coverage'},
							{'key':'SRC_COVERAGE_DIR', 'value':'coverage_report'},
				]
			  }
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		r = requests.post(builds_url, headers=headers, verify=True, json=payload)		  
	if not r.ok:
		print "Failed:", r.content
		sys.exit()
	data = r.json()
	try:
		print "Build %s status: %s" % (data['id'], data['status'])
	except Exception:
		print "Failed:", r.content
