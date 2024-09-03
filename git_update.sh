#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add an example'

git push origin master

echo '------- update complete --------'