#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add nomenclature'

git push origin master

echo '------- update complete --------'