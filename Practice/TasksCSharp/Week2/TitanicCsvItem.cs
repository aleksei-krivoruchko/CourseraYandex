using CsvHelper.Configuration;

namespace TasksCSharp.Week2
{
    //PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    public class TitanicCsvItem
    {
        public string PassengerId { get; set; }
        public string Survived { get; set; }
        public int Pclass { get; set; }
        public string Name { get; set; }
        public string Sex { get; set; }
        public bool IsMale
        {
            get { return Sex == "male"; }
        }
        public double? Age { get; set; }
        public string SibSp { get; set; }
        public string Parch { get; set; }
        public string Ticket { get; set; }
        public double Fare { get; set; }
        public string Cabin { get; set; }
        public string Embarked { get; set; }
    }

    public sealed class TitanicCsvItemMap : CsvClassMap<TitanicCsvItem>
    {
        public TitanicCsvItemMap()
        {
            Map(m => m.PassengerId).Index(0);
            Map(m => m.Survived).Index(1);
            Map(m => m.Pclass).Index(2);
            Map(m => m.Name).Index(3);
            Map(m => m.Sex).Index(4);
            Map(m => m.Age).Index(5);
            Map(m => m.SibSp).Index(6);
            Map(m => m.Parch).Index(7);
            Map(m => m.Ticket).Index(8);
            Map(m => m.Fare).Index(9);
            Map(m => m.Cabin).Index(10);
            Map(m => m.Embarked).Index(11);
        }
    }
}