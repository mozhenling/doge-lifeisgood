#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'fix a bug'

git push origin master

echo '------- update complete --------'