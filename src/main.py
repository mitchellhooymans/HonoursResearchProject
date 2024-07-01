## This is the script that will run the codebase. Ideally this script should be run,
## with certain parameters to get a particular output.




def run_analysis():
    """Main function to orchestrate the entire analysis pipeline."""

    # Set this up to run the workflow on the first set of things
    # essentially this should be a general program that generate outputs and 
    # we don't have to worry about the specifics of what we are doing
    # 
    
    
    # Design this so far with the intention of doing it with the inital theoeretical GALSEDATLAS + SWIRE Templates
    
    # 1. Choose Models
    
    # 1.1 SED Models - Place Holders for now
    galaxy_templates = 'GALSEDATLAS'
    agn_templates = 'SKIRTOR'
    
    # 1.2 Choice of Colour Space - This will determine the colour space we are investigating.
    colourspace = 'UVJ'

    # 2. Load and preprocess data - Loads all data into the worksapce, models, astronomical colours.
    data = load_preprocess_data(galaxy_templates, agn_templates, colourspace)

    # 3. Run Model Code - Creates composite models and the resulting astromical colours

    # 4. Visualize results - plot the colours on a graph

    # 5. Save results (optional) - Output anything as required.
   
    # ...


if __name__ == "__main__":
    main()