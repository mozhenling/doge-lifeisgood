#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'new benchmark'

git push origin master

echo '------- update complete --------'