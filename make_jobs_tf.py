import os, sys

head_file = sys.argv[1]
jobs = sys.argv[2]

head = []
for h in open(head_file, 'r').readlines():
  h = h.strip()
  head.append(h)

count = 0
for j in open(jobs, 'r').readlines():
  job_name = 'job' + str(count) + '.sh'
  count += 1
  out = open(job_name, 'w')
  for h in head:
    out.write(h + '\n')
  out.write('\n' + j)
