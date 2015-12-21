function args=fsc_initial(args)
args.Weight=lstm_setup(args.layer);
args.Mom.Weight=lstm_mom_setup(args.layer);
end