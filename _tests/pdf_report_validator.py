import glob

pdfs = glob.glob("pdfs/*.pdf")

if len(pdfs) < 1:
    print(f'Too few files. You should have 1 joint final report. (You have {len(pdfs)} in total.)')
    exit()
elif len(pdfs) > 1:
    print(f'Too many files. You should only have 1 final report for your group. (You have {len(pdfs)} in total.)')
    exit()

print('The final report exists.')
