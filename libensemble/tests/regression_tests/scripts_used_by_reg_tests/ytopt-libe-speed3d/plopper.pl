#!/usr/bin/perl
#Change this path!
#Author: Xingfu Wu
# MCS, ANL
# plopper.pl: process the file plopper.py to change the proper app_timeout
#

$A_FILE = "timeoutfile.txt";

$filename1 =  $ARGV[0];
    #print "Start to process ", $filename, "...\n";
    $fname = ">" . $A_FILE;
    open(OUTFILE, $fname);
    open (TEMFILE, $filename1);
    while (<TEMFILE>) {
        $line = $_;
	chomp ($line);

        if ($line =~ /app_timeout =/) {
                ($v1, $v2) = split('= ', $line);
		print OUTFILE $v1, " = ", $ARGV[1], "\n";
	} else {
                print OUTFILE $line, "\n";
        }
    }
   close(TEMFILE);
   close(OUTFILE);
   system("mv $A_FILE $filename1");
#exit main
exit 0;
